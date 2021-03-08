# Unet from "Hundred Layers Tiramisu" paper for document layout segmentation

Vanilla UNet replaced with Dense Blocks

**Label-info**

- body text - yellow
- heading - pink
- caption - green
- header&footer - red
- graphic - light blue
- table - dark_blue

**Logs**

![Train-logs](assets/Train%20Loss.png)
![Test-logs](assets/Test%20Loss.png)

# Results

- IOU - ??
- Accuracy - ??
- Memory Requirements and Inference Speeds - ??

**Visual Results** 

Actual Image            |  Segmented Image
:-------------------------:|:-------------------------:
<img width="300" height="500" src="assets/0.jpg"> | <img width="300" height="500" src="assets/0-mask.jpg">
<img width="300" height="500" src="assets/1.jpg"> | <img width="300" height="500" src="assets/1-mask.jpg">
<img width="300" height="500" src="assets/2.jpg"> | <img width="300" height="500" src="assets/2-mask.jpg">
<img width="300" height="500" src="assets/3.jpg"> | <img width="300" height="500" src="assets/3-mask.jpg">
<img width="300" height="500" src="assets/1.jpg"> | <img width="300" height="500" src="assets/1-table.jpg">

*Last row with the table class!*

