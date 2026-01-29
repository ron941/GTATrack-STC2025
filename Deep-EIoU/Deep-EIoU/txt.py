def convert_yolo_to_mot(yolo_txt_path, output_txt_path):
    with open(yolo_txt_path, 'r') as infile, open(output_txt_path, 'w') as outfile:
        for line in infile:
            # 支援逗號或空白分隔
            parts = line.strip().replace(',', ' ').split()
            if len(parts) < 7:
                continue  # 略過格式不完整行
            frame = parts[0]
            x = parts[2]
            y = parts[3]
            w = parts[4]
            h = parts[5]
            conf = parts[6]
            new_line = f"{frame},-1,{x},{y},{w},{h},{conf},-1,-1,-1\n"
            outfile.write(new_line)
    print(f"✅ 轉換完成，已輸出至 {output_txt_path}")

# ✅ 修改路徑為你的實際路徑
convert_yolo_to_mot(
    "/ssd1/ron_soccer/ultralytics-main/output/128057.txt",
    "/ssd1/ron_soccer/Deep-EIoU/Deep-EIoU/data/football/det1/det2"
)
