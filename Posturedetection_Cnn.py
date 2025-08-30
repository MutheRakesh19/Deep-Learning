import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

class PersonPostureDetector:
    def __init__(self, confidence_threshold=0.7, device=None):
        """
        Initialize the Person Posture Detection system
        """
        self.confidence_threshold = confidence_threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained Faster R-CNN model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Transform for preprocessing
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def estimate_posture(self, bbox, frame_height):
        """Estimate standing/sitting from bounding box aspect ratio"""
        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1
        aspect_ratio = box_height / box_width if box_width > 0 else 0
        bbox_bottom = y2 / frame_height
        
        if aspect_ratio > 1.8 and bbox_bottom > 0.3:
            return 'standing'
        elif aspect_ratio < 1.5:
            return 'sitting'
        else:
            if bbox_bottom > 0.7 and aspect_ratio > 1.4:
                return 'standing'
            else:
                return 'sitting'
    
    def detect_frame(self, frame):
        """Detect persons and their postures in a frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        detections = []
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            if label == 1 and score >= self.confidence_threshold:
                posture = self.estimate_posture(box, frame.shape[0])
                detections.append({'bbox': box, 'confidence': score, 'posture': posture})
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and posture labels"""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox'].astype(int)
            color = (0, 255, 0) if det['posture'] == 'standing' else (255, 0, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label_text = f"{det['posture']} ({det['confidence']:.2f})"
            cv2.putText(annotated, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return annotated
    
    def process_video(self, video_path, output_path=None, display=True):
        """Run detection on a video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = self.detect_frame(frame)
            annotated = self.draw_detections(frame, detections)
            
            if display:
                cv2.imshow("Posture Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if out:
                out.write(annotated)
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print("Processing finished!")

# ---------------------------
# USAGE: Put your video path here
# ---------------------------
if __name__ == "__main__":
    video_path = r"C:\web devlopment\AdobeStock_265308825_Video_HD_Preview.mp4"  # <-- change here
    output_path = "output_posture.mp4"  # optional output
    
    detector = PersonPostureDetector(confidence_threshold=0.7)
    detector.process_video(video_path, output_path=output_path, display=True)

