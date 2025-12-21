from ultralytics import YOLO

def main():
    model=YOLO("yolo11n.pt")

    model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    scale=0.5,
    degrees=90,   
    cls=1.5,
    mosaic=1.0,
    device=0
)

if __name__ == "__main__":
    main()
