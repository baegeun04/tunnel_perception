#!/usr/bin/env python3
"""
posture_judge.py — 자세 판정 로직 (ROS2 무관 · 순수 파이썬)

`/detections` V1 계약 (2026-08-17 확정) 구현.
카메라·ROS2 없이 단위 테스트가 가능하도록 의도적으로 분리했다.

판정 규칙 (N7):
    valid  = 관찰 창에서 자세 판정이 가능했던 프레임 수 (unknown 제외 = 기권)
    fallen = valid 중 "쓰러짐"으로 판정된 프레임 수

    확정:  valid >= MIN_VALID  AND  fallen/valid >= CONFIRM_RATIO
    해제:  valid >= MIN_VALID  AND  fallen/valid <= RELEASE_RATIO
    valid < MIN_VALID          →  person_unknown (확정도 해제도 하지 않음)

발행값 (N2 열거값, N3 소문자):
    person_fallen / person_ok / person_unknown

층위 구분 (N7-a):
    - 추론 자체 실패      → 이 모듈이 아니라 노드가 프레임 미발행 처리 (N1)
    - 추론 성공 + 판정 불가 → person_unknown  ← 이 모듈의 책임
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# COCO 17 keypoint 인덱스
# ---------------------------------------------------------------------------
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
TORSO_KPTS = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)

# N2 허용 열거값 — 이 목록 밖의 값은 발행하지 않는다
CLASS_PERSON_FALLEN = "person_fallen"
CLASS_PERSON_OK = "person_ok"
CLASS_PERSON_UNKNOWN = "person_unknown"

ALLOWED_CLASS_NAMES = frozenset({
    "person_fallen", "person_ok", "person_unknown", "fire", "smoke",
})


def normalize_class_name(raw: str) -> str:
    """N3 — 발행 직전 소문자 정규화. 학습 라벨(Fire/Human)은 그대로 두고 여기서만 변환."""
    return raw.strip().lower()


def is_allowed_class(name: str) -> bool:
    """N2 — 열거값 검사. 정규화 이후에 호출할 것."""
    return name in ALLOWED_CLASS_NAMES


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
@dataclass
class PostureConfig:
    # 각도 임계값
    angle_thr: float = 60.0            # 중력 보정이 적용된 상태
    angle_thr_fallback: float = 70.0   # N8-d: TF 없음/stale 시 보수적으로 상향
    angle_max: float = 120.0           # 이 값 이상은 물리적으로 이상 → 판정 불가
    ratio_thr: float = 2.0             # 보조 신호 (bbox 가로/세로)
    kp_conf: float = 0.5               # 몸통 keypoint 신뢰도 하한

    # 투표 (N7-b)
    window: int = 10
    min_valid: int = 4
    confirm_ratio: float = 0.6
    release_ratio: float = 0.3

    # 시간 (N7-d, N9-c)
    observe_timeout: float = 3.0       # 이 시간 동안 valid 미달이면 지속 WARN
    track_timeout: float = 3.0         # 트랙 미갱신 시 상태 삭제


# ---------------------------------------------------------------------------
# 기하 계산
# ---------------------------------------------------------------------------
def _midpoint(p1, p2) -> Tuple[float, float]:
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def torso_angle(
    kpts: Sequence[Sequence[float]],
    kconf: Sequence[float],
    kp_thr: float,
    gravity: Tuple[float, float] = (0.0, 1.0),
) -> Optional[float]:
    """
    어깨 중점 -> 엉덩이 중점 벡터가 '아래 방향'에서 몇 도 기울었는지 (도).

        0도  = 직립 (엉덩이가 어깨 바로 아래)
        90도 = 수평 (누움)

    gravity: 이미지 평면에 투영된 중력 방향 단위벡터 (N8).
             TF를 못 쓰면 (0, 1) = 이미지 y축 아래 방향으로 폴백한다.
    반환 None = 판정 불가 (몸통 keypoint 신뢰도 미달)
    """
    if any(kconf[i] < kp_thr for i in TORSO_KPTS):
        return None

    sh = _midpoint(kpts[L_SHOULDER], kpts[R_SHOULDER])
    hp = _midpoint(kpts[L_HIP], kpts[R_HIP])

    tx, ty = hp[0] - sh[0], hp[1] - sh[1]
    tn = math.hypot(tx, ty)
    if tn < 1e-6:
        return None

    gx, gy = gravity
    gn = math.hypot(gx, gy)
    if gn < 1e-6:
        return None

    cos_a = (tx * gx + ty * gy) / (tn * gn)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


# ---------------------------------------------------------------------------
# 트랙 단위 상태 (투표 기계)
# ---------------------------------------------------------------------------
@dataclass
class TrackState:
    cfg: PostureConfig
    history: deque = field(default_factory=deque)   # 1=fallen, 0=upright, None=unknown
    confirmed: bool = False
    first_seen: float = 0.0
    last_seen: float = 0.0
    fall_since: Optional[float] = None

    def __post_init__(self):
        self.history = deque(maxlen=self.cfg.window)

    def update(self, lying: Optional[bool], now: float) -> dict:
        """
        lying: True=쓰러짐 / False=아님 / None=판정 불가(기권)
        """
        if self.first_seen == 0.0:
            self.first_seen = now
        self.last_seen = now
        self.history.append(None if lying is None else (1 if lying else 0))

        valid_frames = [v for v in self.history if v is not None]
        valid = len(valid_frames)
        fallen = sum(valid_frames)
        ratio = (fallen / valid) if valid else 0.0

        enough = valid >= self.cfg.min_valid

        if enough:
            if not self.confirmed and ratio >= self.cfg.confirm_ratio:
                self.confirmed = True
                self.fall_since = now
            elif self.confirmed and ratio <= self.cfg.release_ratio:
                self.confirmed = False
                self.fall_since = None
        # enough 가 아니면 확정도 해제도 하지 않는다 (N7-b)

        # 발행값 결정 (N7-c)
        if self.confirmed:
            class_name = CLASS_PERSON_FALLEN
        elif not enough:
            class_name = CLASS_PERSON_UNKNOWN
        else:
            class_name = CLASS_PERSON_OK

        # N9-c: 3초 넘게 표본이 안 차면 지속 미상 — /diagnostics WARN 대상
        persistent_unknown = (
            not enough and (now - self.first_seen) >= self.cfg.observe_timeout
        )

        return {
            "class_name": class_name,
            "confirmed": self.confirmed,
            "valid": valid,
            "fallen": fallen,
            "ratio": ratio,
            "enough": enough,
            "persistent_unknown": persistent_unknown,
            "fall_duration": 0.0 if self.fall_since is None else now - self.fall_since,
        }


# ---------------------------------------------------------------------------
# 다중 인원 판정기
# ---------------------------------------------------------------------------
class PostureJudge:
    """트랙 ID별로 TrackState를 관리한다. 트랙 ID가 바뀌면 새 이력으로 시작(N7-d)."""

    def __init__(self, cfg: Optional[PostureConfig] = None):
        self.cfg = cfg or PostureConfig()
        self._tracks: dict = {}
        # 장착 방향·TF 오류 감지용. 이 값이 계속 오르면 카메라가 기울어졌거나
        # 뒤집힌 것이다 (2026-08-17 실측에서 실제로 발생).
        self.implausible_frames = 0

    # -- 프레임 단위 판정 ---------------------------------------------------
    def _frame_decision(self, angle: Optional[float], ar: float,
                        gravity_valid: bool) -> Tuple[Optional[bool], bool]:
        """
        반환 (판정, 각도이상여부)
            판정 True=쓰러짐 / False=아님 / None=판정 불가

        각도를 못 구하면 종횡비로 대체하지 않는다 — 몸통이 안 보이는 상황은
        종횡비도 못 믿는 상황이기 때문 (초기 구현의 클로즈업 오탐 원인).

        angle_max 상한 (2026-08-17 실측 반영):
            엉덩이가 어깨보다 뚜렷이 위에 있는 상태(>120도)는 사람이 누운 자세가
            아니라 카메라 장착 방향 오류·TF 오류·pose 오검출일 가능성이 높다.
            실측에서 카메라가 뒤집힌 채로 ang=172도가 나왔고, 그것이 조용히
            person_fallen 으로 확정되었다.
            이 값들은 person_ok 로도 person_fallen 으로도 보내지 않고
            판정 불가(unknown)로 돌린다. 근본 해결은 N8 중력 보정이며
            이 검사는 오판을 막는 방어장치일 뿐이다.
        """
        if angle is None:
            return None, False
        if angle >= self.cfg.angle_max:
            return None, True
        thr = self.cfg.angle_thr if gravity_valid else self.cfg.angle_thr_fallback
        return ((angle >= thr) or (ar >= self.cfg.ratio_thr)), False

    def update(
        self,
        track_id,
        kpts: Sequence[Sequence[float]],
        kconf: Sequence[float],
        bbox: Tuple[float, float, float, float],   # x1, y1, x2, y2
        gravity: Optional[Tuple[float, float]] = None,
        now: Optional[float] = None,
    ) -> dict:
        """
        gravity=None  → TF 미사용/실패. 이미지 y축 폴백 + 임계값 상향 (N8-d)
        """
        now = time.time() if now is None else now
        gravity_valid = gravity is not None
        g = gravity if gravity_valid else (0.0, 1.0)

        x1, y1, x2, y2 = bbox
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        ar = bw / bh

        angle = torso_angle(kpts, kconf, self.cfg.kp_conf, g)
        lying, implausible = self._frame_decision(angle, ar, gravity_valid)
        if implausible:
            self.implausible_frames += 1

        st = self._tracks.get(track_id)
        if st is None:
            st = TrackState(cfg=self.cfg)
            self._tracks[track_id] = st

        out = st.update(lying, now)
        out.update({
            "track_id": track_id,
            "angle": angle,
            "aspect_ratio": ar,
            "gravity_valid": gravity_valid,
            "angle_thr": self.cfg.angle_thr if gravity_valid
                         else self.cfg.angle_thr_fallback,
            "implausible_angle": implausible,
        })
        return out

    def prune(self, now: Optional[float] = None) -> int:
        """오래 안 보인 트랙 삭제. 반환 = 삭제된 개수."""
        now = time.time() if now is None else now
        stale = [k for k, v in self._tracks.items()
                 if now - v.last_seen > self.cfg.track_timeout]
        for k in stale:
            del self._tracks[k]
        return len(stale)

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)


# ---------------------------------------------------------------------------
# /diagnostics 카운터 (N5) — 3방향
# ---------------------------------------------------------------------------
@dataclass
class ModelAgreementCounters:
    pose_miss_human_hit: int = 0     # pose 0명 · Human >=1  (역할 A 요청 항목)
    pose_hit_human_miss: int = 0     # pose >=1 · Human 0    (pose 과검출 추정)
    both_hit_count_differ: int = 0   # 양쪽 검출됐으나 인원수 불일치
    frames_total: int = 0

    def update(self, pose_count: int, human_count: int) -> None:
        self.frames_total += 1
        if pose_count == 0 and human_count >= 1:
            self.pose_miss_human_hit += 1
        elif pose_count >= 1 and human_count == 0:
            self.pose_hit_human_miss += 1
        elif pose_count >= 1 and human_count >= 1 and pose_count != human_count:
            self.both_hit_count_differ += 1

    def as_dict(self) -> dict:
        return {
            "pose_miss_human_hit": self.pose_miss_human_hit,
            "pose_hit_human_miss": self.pose_hit_human_miss,
            "both_hit_count_differ": self.both_hit_count_differ,
            "frames_total": self.frames_total,
        }
