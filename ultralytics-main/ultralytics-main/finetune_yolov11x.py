from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("/ssd1/ron_soccer/ultralytics-main/best.pt")

    model.train(
        data="/ssd1/ron_soccer/ultralytics-main/data/final.yaml",
        device=1,
        epochs=60,            
        imgsz=1280,                
        batch=8,                   
        lr0=0.003,                 
        freeze=0,
        optimizer='AdamW',
        patience=50,               
        cos_lr=True,               
        dropout=0.1,              
        rect=True,
        mosaic=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mixup=0.0,
        degrees=2,
        translate=0.05,
        scale=0.15,               
        shear=0.0,
        weight_decay=0.0001,       
        label_smoothing=0.05,     
        multi_scale=True,           
        project="/ssd1/ron_soccer/ultralytics-main/runs_finetune",
        name="finetune_result",        
    )

  
    try:
        model.save("/ssd1/ron_soccer/ultralytics-main/finetune_best.pt")
        print("✅ 模型已手動儲存為 finetune_best.pt")
    except Exception as e:
        print(f"❌ 模型儲存失敗: {e}")
