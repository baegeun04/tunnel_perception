#!/bin/bash
# 인지 스택 한 줄 기동/종료 (역할 B → 역할 A 인수용)
# 사용: bash ~/p.sh          (기동)
#      bash ~/p.sh stop      (종료)
#      bash ~/p.sh status    (상태 확인)

LOGDIR=/tmp/percep_logs
mkdir -p "$LOGDIR"

PIDS=(camera.pid perception.pid thermal_pub.pid thermal_blob.pid adapter.pid)

stop_all() {
    echo "=== 인지 스택 내리는 중 ==="
    for f in "${PIDS[@]}"; do
        if [ -f "$LOGDIR/$f" ]; then
            PID=$(cat "$LOGDIR/$f")
            if kill -0 "$PID" 2>/dev/null; then
                kill -9 "$PID" 2>/dev/null
                echo "  killed $f (pid $PID)"
            fi
            rm -f "$LOGDIR/$f"
        fi
    done
    pkill -9 -f "component_container.*camera_container" 2>/dev/null
    sleep 2
    echo "=== 완료 (시리얼 포트 반납 대기 2초 포함) ==="
}

status_all() {
    echo "=== /scan (라이다, 역할 A 쪽 — 참고용) ==="
    timeout 3 ros2 topic hz /scan 2>&1 | head -3
    echo ""
    echo "=== /camera/color/image_raw ==="
    timeout 5 ros2 topic hz /camera/color/image_raw --window 5 2>&1 | tail -3
    echo ""
    echo "=== /detections ==="
    timeout 5 ros2 topic hz /detections --window 5 2>&1 | tail -3
    echo ""
    echo "=== /thermal/temp_raw ==="
    timeout 5 ros2 topic hz /thermal/temp_raw --window 5 2>&1 | tail -3
    echo ""
    echo "=== /thermal/person_present (최근 1개) ==="
    timeout 3 ros2 topic echo /thermal/person_present --once 2>&1
    echo ""
    echo "=== /adapter_status (최근 1개) ==="
    timeout 3 ros2 topic echo /adapter_status --once 2>&1
}

if [ "$1" = "stop" ]; then
    stop_all
    exit 0
fi

if [ "$1" = "status" ]; then
    status_all
    exit 0
fi

echo "=== 인지 스택 기동 시작 ==="

source /opt/ros/humble/setup.bash
source ~/percep_ws/install/setup.bash

# 1단계: 카메라 — 물리적으로 똑바로 세워서 연결된 상태여야 함
echo "[1/5] 카메라 기동..."
nohup ros2 launch orbbec_camera gemini2.launch.py \
    > "$LOGDIR/camera.log" 2>&1 &
echo $! > "$LOGDIR/camera.pid"

# 카메라가 실제로 뜰 때까지 대기 (최대 20초, /camera/color/image_raw 확인)
for i in $(seq 1 20); do
    if timeout 2 ros2 topic hz /camera/color/image_raw > /dev/null 2>&1; then
        echo "  카메라 준비됨 (${i}초)"
        break
    fi
    sleep 1
done

# 2단계: perception_node — ros2 run 대신 python3 직접 호출 (yolo_env shebang 문제 회피)
echo "[2/5] perception_node 기동..."
source ~/yolo_env/bin/activate
nohup python3 ~/percep_ws/install/vision_pipeline/lib/vision_pipeline/perception_node \
    > "$LOGDIR/perception.log" 2>&1 &
echo $! > "$LOGDIR/perception.pid"
deactivate

for i in $(seq 1 30); do
    if timeout 2 ros2 topic hz /detections > /dev/null 2>&1; then
        echo "  perception_node 준비됨 (${i}초)"
        break
    fi
    sleep 1
done

# 3단계: 열화상 publisher — yolo_env 끄고 시스템 python3 로
echo "[3/5] thermal_publisher 기동..."
nohup python3 ~/percep_ws/install/thermal_camera/lib/thermal_camera/thermal_publisher \
    --ros-args -p port:=/dev/thermal -p baud:=115200 \
    > "$LOGDIR/thermal_pub.log" 2>&1 &
echo $! > "$LOGDIR/thermal_pub.pid"
sleep 3

# 4단계: 열화상 blob 판정
echo "[4/5] thermal_blob_node 기동..."
nohup python3 ~/percep_ws/install/vision_pipeline/lib/vision_pipeline/thermal_blob_node \
    > "$LOGDIR/thermal_blob.log" 2>&1 &
echo $! > "$LOGDIR/thermal_blob.pid"
sleep 2

# 5단계: 어댑터 — TF 인자 필수 (camera_frame:=camera_link optical:=false)
#   이유: Orbbec 드라이버가 이미 base_link 하위 전체 TF 체인을 발행하므로
#         기본값(camera_color_optical_frame, optical:=true)으로 켜면
#         같은 프레임에 부모가 둘 생겨 TF 트리가 깨짐
echo "[5/5] adapter 기동..."
source ~/ros2_ws/install/setup.bash
nohup ros2 launch perception_adapter adapter.launch.py \
    cam_x:=0.25 cam_z:=0.55 camera_frame:=camera_link optical:=false \
    > "$LOGDIR/adapter.log" 2>&1 &
echo $! > "$LOGDIR/adapter.pid"

echo ""
echo "=== 기동 완료. 상태 확인: bash ~/p.sh status ==="
echo "=== 로그 위치: $LOGDIR/*.log ==="
