import cv2 as cv
import easyocr

# 1. Khởi tạo Reader (Dùng CPU vì máy bạn báo không có CUDA)
reader = easyocr.Reader(['en'], gpu=False) 

video_path = "plate2.mp4"
cap = cv.VideoCapture(video_path)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Tiền xử lý để tìm khung hình (Thực hiện mỗi frame để box mượt mà)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(blur, 30, 150)
    
    # Tìm các đường bao (contours)
    contours, _ = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    img_size = frame.shape[0] * frame.shape[1]

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = w / h
        area_ratio = (w * h) / img_size
        
        # Lọc các khung hình có tỉ lệ giống biển số xe (thường là hình chữ nhật nằm ngang)
        if (2.0 < aspect_ratio < 5.5) and (0.0001 < area_ratio < 0.02):
            
            # --- VẼ BOX LÊN MÀN HÌNH ---
            # Vẽ hình chữ nhật màu xanh lá (0, 255, 0), độ dày nét vẽ là 2
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Chỉ chạy OCR mỗi 10 frame để tránh giật lag video
            if frame_count % 10 == 0:
                plate_roi = gray[y:y+h, x:x+w]
                result = reader.readtext(plate_roi, detail=0)
                
                if result:
                    text = result[0].upper()
                    # Hiển thị text ngay trên đầu cái box
                    cv.putText(frame, text, (x, y - 10), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    print(f"Detected: {text}")

    # Hiển thị kết quả ra màn hình
    cv.imshow("Tracking License Plate", frame)

    # Thoát nếu nhấn 'q' hoặc ESC
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()