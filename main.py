import torch
from PIL import Image
import cv2
from pathlib import Path
import pandas as pd

# Model
model = torch.hub.load(
    "yolov5", "yolov5s", source="local"
)  # or yolov5n - yolov5x6, custom

# Images
img = "test.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
print(results.pandas())  # or .show(), .save(), .crop(), .pandas(), etc.
# Convert results to pandas DataFrame
# df_results = pd.DataFrame(
#     columns=["Class", "Score", "X_min", "Y_min", "X_max", "Y_max"]
# )

# for det in results.pred[0]:
#     label = int(det[-1])
#     score = det[4].item()
#     bbox = det[:4].cpu().numpy()
#     df_results = df_results.assign(
#         {
#             "Class": label,
#             "Score": score,
#             "X_min": bbox[0],
#             "Y_min": bbox[1],
#             "X_max": bbox[2],
#             "Y_max": bbox[3],
#         },
#         ignore_index=True,
#     )

# # Display DataFrame
# print(df_results)

# # Get detected bounding boxes, labels, and scores
# boxes = results.xyxy[0].cpu().numpy()
# labels = results.pred[0][:, -1].cpu().numpy()
# scores = results.pred[0][:, 4].cpu().numpy()

# # Load image with OpenCV
# img_cv2 = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

# # Draw bounding boxes on the image
# # for box, label, score in zip(boxes, labels, scores):
# #     x_min, y_min, x_max, y_max = map(int, box)
# #     label = int(label)
# #     cv2.rectangle(img_cv2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
# #     cv2.putText(
# #         img_cv2,
# #         f"Class: {label}, Score: {score:.2f}",
# #         (x_min, y_min - 10),
# #         cv2.FONT_HERSHEY_SIMPLEX,
# #         0.5,
# #         (0, 255, 0),
# #         2,
# #     )

# # Display the image with bounding boxes
# cv2.imshow("YOLOv5 Object Detection", img_cv2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
