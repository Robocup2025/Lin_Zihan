import cv2
from ultralytics import YOLO

# 0. 框颜色设置
COLORS=[(0,255,0),    
        (255,0,0),    
        (0,0,255),    
        (255,255,0),  
        (255,0,255), 
        (0,255,255), 
        (128,255,0), 
        (0,128,255)]

# 1. 路径设置
MODEL_PATH="best.pt"
VIDEO_IN="test_positive.mp4" #另外两个test_revolve.mp4,test_comprehension.mp4
VIDEO_OUT="output_positive.mp4" #另外两个output_revolve.mp4,output_comprehension.mp4

# 2. 加载模型
model=YOLO(MODEL_PATH)

# 3. 打开视频
cap=cv2.VideoCapture(VIDEO_IN)
fps=cap.get(cv2.CAP_PROP_FPS)
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc=cv2.VideoWriter_fourcc(*"mp4v")
writer=cv2.VideoWriter(VIDEO_OUT,fourcc,fps,(width,height))

# 4. 逐帧检测
while True:
    ret,frame=cap.read()
    if not ret:
        break

    results=model(frame,conf=0.3,verbose=False)

    for r in results:
        boxes=r.boxes
        names=r.names

        for box in boxes:
            cls_id=int(box.cls[0])
            label=names[cls_id]
            conf=float(box.conf[0])

            x1,y1,x2,y2=map(int,box.xyxy[0])

            # 4.1 取对应颜色
            color=COLORS[cls_id]

            # 4.2 终端打印
            print(f"{label}:({x1},{y1})-({x2},{y2}),conf={conf:.2f}")

            # 4.3 画框+类别+置信度
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"{label} {conf:.2f}",(x1,max(20,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    writer.write(frame)

    cv2.imshow("YOLOv11 Detection",frame)
    if cv2.waitKey(1) & 0xFF==27:
        break

# 5. 释放资源
cap.release()
writer.release()
cv2.destroyAllWindows()

print("检测完成")
