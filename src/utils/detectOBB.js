import * as tf from "@tensorflow/tfjs";
import { renderBoxes } from "./renderBox";

// Default class names for DOTA dataset
// These will be overridden by metadata when available
let CLASS_NAMES = [
  "plane", "ship", "storage tank", "baseball diamond", "tennis court",
  "basketball court", "ground track field", "harbor", "bridge", "large vehicle",
  "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool"
];

// Default image size (will be overridden by metadata when available)
let MODEL_IMAGE_SIZE = [1024, 1024];

// Function to update class names and image size from metadata
export const updateModelConfig = (metadata) => {
  if (metadata) {
    // Update class names if available in metadata
    if (metadata.names) {
      const nameEntries = Object.entries(metadata.names).sort(([a,], [b,]) => parseInt(a) - parseInt(b));
      CLASS_NAMES = nameEntries.map(([_, name]) => name);
      console.log("Updated class names from metadata:", CLASS_NAMES);
    }
    
    // Update image size if available in metadata
    if (metadata.imgsz && Array.isArray(metadata.imgsz) && metadata.imgsz.length >= 2) {
      MODEL_IMAGE_SIZE = [metadata.imgsz[0], metadata.imgsz[1]];
      console.log("Updated model image size from metadata:", MODEL_IMAGE_SIZE);
    }
  }
};

// 将xywhr格式转换为4个角点坐标
const convertToCorners = (cx, cy, w, h, angle) => {
  // 计算旋转后的四个角点
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);
  
  // 计算四个角点相对于中心的偏移
  const dx1 = -w/2, dy1 = -h/2;
  const dx2 = w/2, dy2 = -h/2;
  const dx3 = w/2, dy3 = h/2;
  const dx4 = -w/2, dy4 = h/2;
  
  // 应用旋转变换
  return [
    cx + cosA * dx1 - sinA * dy1, cy + sinA * dx1 + cosA * dy1, // 左上
    cx + cosA * dx2 - sinA * dy2, cy + sinA * dx2 + cosA * dy2, // 右上
    cx + cosA * dx3 - sinA * dy3, cy + sinA * dx3 + cosA * dy3, // 右下
    cx + cosA * dx4 - sinA * dy4, cy + sinA * dx4 + cosA * dy4, // 左下
  ];
};

// 计算两个有向边界框之间的IoU
const calculateOBBIoU = (box1Corners, box2Corners) => {
  // 将角点坐标转换为多边形格式
  const polygon1 = [];
  const polygon2 = [];
  
  for (let i = 0; i < 4; i++) {
    polygon1.push({x: box1Corners[i*2], y: box1Corners[i*2+1]});
    polygon2.push({x: box2Corners[i*2], y: box2Corners[i*2+1]});
  }
  
  // 使用Sutherland-Hodgman算法计算多边形相交
  const intersection = clipPolygons(polygon1, polygon2);
  
  if (!intersection || intersection.length === 0) {
    return 0;
  }
  
  // 计算多边形面积
  const area1 = calculatePolygonArea(polygon1);
  const area2 = calculatePolygonArea(polygon2);
  const intersectionArea = calculatePolygonArea(intersection);
  
  // 计算IoU
  return intersectionArea / (area1 + area2 - intersectionArea);
};

// 计算多边形面积
const calculatePolygonArea = (polygon) => {
  let area = 0;
  const n = polygon.length;
  
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    area += polygon[i].x * polygon[j].y;
    area -= polygon[j].x * polygon[i].y;
  }
  
  return Math.abs(area) / 2;
};

// 判断点是否在多边形内部
const isPointInPolygon = (point, polygon) => {
  let inside = false;
  const n = polygon.length;
  
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygon[i].x, yi = polygon[i].y;
    const xj = polygon[j].x, yj = polygon[j].y;
    
    const intersect = ((yi > point.y) !== (yj > point.y)) &&
      (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi);
      
    if (intersect) inside = !inside;
  }
  
  return inside;
};

// 切割多边形 (Sutherland-Hodgman 算法)
const clipPolygons = (subjectPolygon, clipPolygon) => {
  let outputList = subjectPolygon;
  
  for (let i = 0; i < clipPolygon.length; i++) {
    const clipEdgeStart = clipPolygon[i];
    const clipEdgeEnd = clipPolygon[(i + 1) % clipPolygon.length];
    
    const inputList = outputList;
    outputList = [];
    
    if (inputList.length === 0) {
      break;
    }
    
    const S = inputList[inputList.length - 1];
    
    for (let j = 0; j < inputList.length; j++) {
      const E = inputList[j];
      
      if (isInside(E, clipEdgeStart, clipEdgeEnd)) {
        if (!isInside(S, clipEdgeStart, clipEdgeEnd)) {
          outputList.push(computeIntersection(S, E, clipEdgeStart, clipEdgeEnd));
        }
        outputList.push(E);
      } else if (isInside(S, clipEdgeStart, clipEdgeEnd)) {
        outputList.push(computeIntersection(S, E, clipEdgeStart, clipEdgeEnd));
      }
    }
  }
  
  return outputList;
};

// 判断点是否在边的内侧
const isInside = (p, edgeStart, edgeEnd) => {
  return (edgeEnd.x - edgeStart.x) * (p.y - edgeStart.y) - 
         (edgeEnd.y - edgeStart.y) * (p.x - edgeStart.x) >= 0;
};

// 计算两条线段的交点
const computeIntersection = (s, e, clipEdgeStart, clipEdgeEnd) => {
  const dc = {x: clipEdgeStart.x - clipEdgeEnd.x, y: clipEdgeStart.y - clipEdgeEnd.y};
  const dp = {x: s.x - e.x, y: s.y - e.y};
  
  const n1 = clipEdgeStart.x * clipEdgeEnd.y - clipEdgeStart.y * clipEdgeEnd.x;
  const n2 = s.x * e.y - s.y * e.x;
  const n3 = 1.0 / (dc.x * dp.y - dc.y * dp.x);
  
  return {
    x: (n1 * dp.x - n2 * dc.x) * n3,
    y: (n1 * dp.y - n2 * dc.y) * n3
  };
};

// 非极大值抑制函数
const applyNMS = (boxes, iouThreshold) => {
  // 按置信度排序
  const sortedBoxes = [...boxes].sort((a, b) => b.confidence - a.confidence);
  const selectedBoxes = [];
  
  while (sortedBoxes.length > 0) {
    // 选择置信度最高的框
    const currentBox = sortedBoxes.shift();
    selectedBoxes.push(currentBox);
    
    // 移除与当前框IoU大于阈值的框
    let i = 0;
    while (i < sortedBoxes.length) {
      const iou = calculateOBBIoU(currentBox.corners, sortedBoxes[i].corners);
      
      // 如果同一类别且IoU大于阈值，则丢弃该框
      if (currentBox.classIndex === sortedBoxes[i].classIndex && iou > iouThreshold) {
        sortedBoxes.splice(i, 1);
      } else {
        i++;
      }
    }
  }
  
  return selectedBoxes;
};

// 主OBB检测函数
export const detectOBB = async (imgSource, model, canvas, confThreshold = 0.35, nmsThreshold = 1e-9) => {
  if (!model.net) return;

  // Use model input shape if available, otherwise use the size from metadata
  const [modelWidth, modelHeight] = model.inputShape ? 
    model.inputShape.slice(1, 3) : 
    MODEL_IMAGE_SIZE;
    
  const img = tf.browser.fromPixels(imgSource);
  
  // 调整图像尺寸以适应模型输入
  const input = tf.tidy(() => {
    const resized = tf.image.resizeBilinear(img, [modelWidth, modelHeight]);
    const normalized = resized.div(255.0);
    return normalized.expandDims(0);
  });

  try {
    // 模型推理
    const outputs = await model.net.executeAsync(input);
    
    // 打印模型原始输出的形状
    if (Array.isArray(outputs)) {
      console.log("Model outputs is an array with shapes:");
      outputs.forEach((tensor, i) => {
        console.log(`  Output[${i}]: ${tensor.shape}`);
      });
    } else {
      console.log(`Model output shape: ${outputs.shape}`);
    }
    
    // 处理模型输出 - 健壮性处理，支持多种输出格式
    let predictions;
    
    // 检查outputs是数组还是单个张量
    if (Array.isArray(outputs)) {
      // 如果是数组，找到包含预测的张量
      // YOLO模型可能会输出多个张量，我们需要找到正确的那个
      if (outputs[0].shape.length === 3) { // [batch, classes+bbox, predictions]
        predictions = outputs[0];
      } else {
        // 尝试其他可能的输出格式
        for (let i = 0; i < outputs.length; i++) {
          if (outputs[i].shape.length === 3) {
            predictions = outputs[i];
            break;
          }
        }
        // 如果仍然没有找到，使用第一个输出
        if (!predictions) {
          predictions = outputs[0];
        }
      }
    } else {
      // 如果是单个张量
      predictions = outputs;
    }

    // 确保我们可以处理张量
    let predsArray;
    if (predictions.shape.length === 3) { // [batch, 20, 21504]
      predsArray = await predictions.arraySync();
      console.log(`Processed predictions shape: ${predictions.shape}`);
      
      if (!predsArray || !predsArray[0]) {
        throw new Error("预测结果格式不正确");
      }
    } else {
      throw new Error("无法解析模型输出格式");
    }

    // 释放资源
    tf.dispose([img, input, ...Array.isArray(outputs) ? outputs : [outputs]]);

    // 处理预测结果
    const ctx = canvas.getContext("2d");
    canvas.width = imgSource.width;
    canvas.height = imgSource.height;
    
    // 清除上一帧
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    
    // 绘制预测框和统计类别数量
    let personCount = 0;
    const boxes = [];
    const threshold = confThreshold; // 使用传入的置信度阈值
    
    // 新增：按类别统计检测数量
    const classCounts = {};
    CLASS_NAMES.forEach(name => {
      classCounts[name] = 0;
    });
    
    console.log(`Model input shape: ${modelWidth}x${modelHeight}`);
    console.log(`Canvas size: ${canvas.width}x${canvas.height}`);
    
    // 计算模型输入尺寸和实际显示尺寸之间的缩放比例
    const scaleX = canvas.width / modelWidth;
    const scaleY = canvas.height / modelHeight;
    console.log(`Scale factors: scaleX=${scaleX}, scaleY=${scaleY}`);
    
    // 处理预测结果
    // YOLOv8-OBB输出通常是[1, 20, 21504]形状
    // 需要转置或重新排列数据进行处理
    const predictions_data = predsArray[0];
    
    // 假设YOLOv8-OBB的输出是[classes+bbox, predictions]格式
    // 我们需要对每个预测点进行处理
    for (let i = 0; i < predictions_data[0].length; i++) {
      // 从不同维度提取数据
      // 根据YOLOv8-OBB的输出格式进行适配
      const data = [];
      for (let j = 0; j < predictions_data.length; j++) {
        data.push(predictions_data[j][i]);
      }
      
      // 现在data包含了单个检测框的所有信息
      // data的前4个元素是[cx, cy, w, h]
      const cx = data[0];
      const cy = data[1];
      const w = data[2];
      const h = data[3];
      
      // 类别置信度从索引4开始
      const classScores = data.slice(4, 4 + CLASS_NAMES.length);
      const classIndex = classScores.indexOf(Math.max(...classScores));
      const confidence = classScores[classIndex];
      
      // 旋转角度通常是最后一个元素
      const angle = data[data.length - 1];
      
      // 过滤掉低置信度检测结果
      if (confidence > threshold) {
        // 更新类别计数
        const className = CLASS_NAMES[classIndex];
        classCounts[className] = (classCounts[className] || 0) + 1;
        
        // 如果是人类类别，增加计数（为了与原代码兼容，虽然人类不在当前类别中）
        if (className === "person") {
          personCount++;
        }
        
        // 将xywhr转换为四个角点坐标
        const corners = convertToCorners(cx* scaleX, cy* scaleY, w* scaleX, h* scaleY, angle);

        // 缩放角点坐标到显示尺寸
        const scaledCorners = [
          corners[0] , corners[1],
          corners[2] , corners[3],
          corners[4] , corners[5],
          corners[6] , corners[7]
        ];
        
        boxes.push({
          debug: {cx, cy, w, h, angle},
          corners: scaledCorners,
          confidence: confidence,
          classIndex: classIndex
        });
      }
    }
    
    console.log("检测到的框数量(NMS前):", boxes.length);
    
    // 应用非极大值抑制
    const iouThreshold = nmsThreshold; // 使用传入的IoU阈值
    const filteredBoxes = applyNMS(boxes, iouThreshold);
    console.log("检测到的框数量(NMS后):", filteredBoxes.length);
    
    // 重新计算类别计数
    Object.keys(classCounts).forEach(key => {
      classCounts[key] = 0;
    });
    
    filteredBoxes.forEach(box => {
      const className = CLASS_NAMES[box.classIndex];
      classCounts[className] = (classCounts[className] || 0) + 1;
    });
    
    // 绘制应用NMS后的OBB框
    drawOBBBoxes(ctx, filteredBoxes);
    
    // 显示检测统计结果
    displayDetectionSummary(ctx, classCounts);
    
    // 查找检测到的总数量
    const totalDetections = Object.values(classCounts).reduce((sum, count) => sum + count, 0);
    console.log(`Number of detections: ${totalDetections}`);
    for (const className in classCounts) {
      if (classCounts[className] > 0) {
        console.log(`  - ${className}: ${classCounts[className]}`);
      }
    }
    
    // 更新人数计数（使用总检测数而非人数，因为当前模型类别中没有person）
    if (model.onCountChange) {
      model.onCountChange(totalDetections);
    }
  } catch (error) {
    console.error("OBB检测处理错误:", error);
    
    // 即使出错也需要清理资源
    tf.dispose([img, input]);
    
    // 可以添加错误反馈到UI，例如在canvas上显示错误信息
    const ctx = canvas.getContext("2d");
    ctx.font = "20px Arial";
    ctx.fillStyle = "red";
    ctx.fillText("检测处理出错: " + error.message, 10, 30);
  }
};

// 显示检测统计结果
const displayDetectionSummary = (ctx, classCounts) => {
  let y = 30;
  const totalDetections = Object.values(classCounts).reduce((sum, count) => sum + count, 0);
  
  ctx.font = "16px Arial";
  ctx.fillStyle = "rgba(0,0,0,0.6)";
  ctx.fillRect(10, y - 20, 260, 25 + Object.keys(classCounts).filter(key => classCounts[key] > 0).length * 20);
  
  ctx.fillStyle = "white";
  ctx.fillText(`检测总数: ${totalDetections}`, 15, y);
  y += 25;
  
  for (const className in classCounts) {
    if (classCounts[className] > 0) {
      ctx.fillText(`${className}: ${classCounts[className]}`, 15, y);
      y += 20;
    }
  }
};

// 视频检测函数
export const detectOBBVideo = (videoElement, model, canvas, confThreshold = 0.35, nmsThreshold = 1e-9) => {
  const detectFrame = async () => {
    if (videoElement.paused || videoElement.ended) return;
    await detectOBB(videoElement, model, canvas, confThreshold, nmsThreshold);
    requestAnimationFrame(detectFrame);
  };
  detectFrame();
};

// 绘制OBB框
const drawOBBBoxes = (ctx, boxes) => {
  // 使用不同颜色区分不同类别
  const colors = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
    "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC"
  ];
  
  boxes.forEach(box => {
    const { corners, confidence, classIndex } = box;
    const classLabel = CLASS_NAMES[classIndex];
    
    // 设置绘图样式 - 使用类别对应的颜色
    ctx.lineWidth = 2;
    ctx.strokeStyle = colors[classIndex % colors.length];
    ctx.fillStyle = colors[classIndex % colors.length];
    
    // 绘制旋转的边界框
    ctx.beginPath();
    ctx.moveTo(corners[0], corners[1]);
    ctx.lineTo(corners[2], corners[3]);
    ctx.lineTo(corners[4], corners[5]);
    ctx.lineTo(corners[6], corners[7]);
    ctx.closePath();
    ctx.stroke();
    
    // 绘制标签
    const text = `${classLabel} ${(confidence * 100).toFixed(1)}%`;
    ctx.font = "12px Arial";
    const textWidth = ctx.measureText(text).width;
    ctx.fillRect(corners[0], corners[1] - 17, textWidth + 4, 17);
    ctx.fillStyle = "#000000";
    ctx.fillText(text, corners[0] + 2, corners[1] - 5);
  });
};
