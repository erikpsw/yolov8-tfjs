import React, { useState, useEffect, useRef, useMemo } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl"; // set backend to webgl
import Loader from "./components/loader";
import ButtonHandler from "./components/btn-handler";
import { detect, detectVideo } from "./utils/detect";
import { detectOBB, detectOBBVideo } from "./utils/detectOBB"; // 导入OBB检测函数
import "./style/App.css";

const App = () => {
  const [loading, setLoading] = useState({ loading: true, progress: 0 }); // loading state
  const [personCount, setPersonCount] = useState(0); // 添加人数计数state
  const [modelName, setModelName] = useState("yolov8n"); // 默认使用yolov8n模型
  const [confThreshold, setConfThreshold] = useState(0.35); // 置信度阈值
  const [nmsExponent, setNmsExponent] = useState(9); // NMS阈值的指数部分
  const [model, setModel] = useState({
    net: null,
    inputShape: [1, 0, 0, 3],
    onCountChange: (count) => setPersonCount(count), // 添加回调函数
  }); // init model & input shape

  // 计算实际的NMS阈值
  const nmsThreshold = useMemo(() => {
    return Math.pow(10, -nmsExponent);
  }, [nmsExponent]);

  // references
  const imageRef = useRef(null);
  const cameraRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const loadModel = async (modelName) => {
    setLoading({ loading: true, progress: 0 });
    
    // 清理旧模型
    if (model.net) {
      model.net.dispose();
    }
    
    try {
      const yolov8 = await tf.loadGraphModel(
        `${window.location.href}/${modelName}_web_model/model.json`,
        {
          onProgress: (fractions) => {
            setLoading({ loading: true, progress: fractions }); // set loading fractions
          },
        }
      ); // load model

      // 获取模型输入形状
      const inputShape = yolov8.inputs[0].shape;
      
      // 根据输入形状创建适当的测试输入
      const dummyInput = tf.ones(inputShape);
      
      try {
        const warmupResults = yolov8.execute(dummyInput);
        tf.dispose(warmupResults); // cleanup memory
        
        setLoading({ loading: false, progress: 1 });
        setModel({
          net: yolov8,
          inputShape: inputShape,
          onCountChange: (count) => setPersonCount(count),
        });
      } catch (execError) {
        console.error("模型执行测试失败:", execError);
        setLoading({ loading: false, progress: 0 });
        alert(`模型 ${modelName} 测试执行失败，请尝试其他模型。错误: ${execError.message}`);
      }

      tf.dispose(dummyInput); // cleanup memory
    } catch (error) {
      console.error("模型加载失败:", error);
      setLoading({ loading: false, progress: 0 });
      alert(`模型 ${modelName} 加载失败，请尝试其他模型。错误: ${error.message}`);
    }
  };

  // 切换模型
  const switchModel = (newModelName) => {
    if (newModelName !== modelName) {
      setModelName(newModelName);
      
      // 重置UI状态和资源
      if (imageRef.current) {
        imageRef.current.src = "#";
        imageRef.current.style.display = "none";
      }
      
      if (videoRef.current) {
        videoRef.current.src = "";
        videoRef.current.style.display = "none";
      }
      
      if (cameraRef.current) {
        cameraRef.current.srcObject = null;
        cameraRef.current.style.display = "none";
      }
      
      // 清除画布
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      
      // 重置人数计数
      setPersonCount(0);
      
      // 加载新模型
      loadModel(newModelName);
    }
  };

  // 重新检测当前图像/视频的函数
  const handleRedetect = () => {
    if (!model.net) return;

    // 检查当前激活的媒体元素
    if (imageRef.current && imageRef.current.style.display !== 'none' && imageRef.current.src !== '#') {
      // 重新检测图像
      const detectFunc = getDetectFunction(true);
      detectFunc(imageRef.current, model, canvasRef.current);
    } 
    else if (videoRef.current && videoRef.current.style.display !== 'none') {
      // 重新检测视频（需要先暂停当前检测循环）
      const detectFunc = getDetectFunction(false);
      detectFunc(videoRef.current, model, canvasRef.current);
    }
    else if (cameraRef.current && cameraRef.current.style.display !== 'none') {
      // 重新检测摄像头（需要先暂停当前检测循环）
      const detectFunc = getDetectFunction(false);
      detectFunc(cameraRef.current, model, canvasRef.current);
    }
  };

  // 根据模型类型选择合适的检测函数
  const getDetectFunction = (isImage = false) => {
    if (modelName.includes("obb")) {
      return isImage ? 
        (source, model, canvas) => detectOBB(source, model, canvas, confThreshold, nmsThreshold) : 
        (source, model, canvas) => detectOBBVideo(source, model, canvas, confThreshold, nmsThreshold);
    }
    return isImage ? 
      (source, model, canvas) => detect(source, model, canvas, confThreshold) : 
      (source, model, canvas) => detectVideo(source, model, canvas, confThreshold);
  };

  useEffect(() => {
    tf.ready().then(() => {
      loadModel(modelName);
    });
  }, []);

  // 当阈值变化时重新检测
  useEffect(() => {
    if (model.net) {
      handleRedetect();
    }
  }, [confThreshold, nmsThreshold]);

  const obb_model = "yolov8n-obb"; // OBB模型名称
  
  return (
    <div className="App">
      {loading.loading && <Loader>模型加载中... {(loading.progress * 100).toFixed(2)}%</Loader>}
      <div className="header">
        <h1>🎯 YOLO 实时人数检测系统</h1>
        <div style={{ fontSize: '24px', color: '#2ecc71', margin: '10px 0' }}>
          当前检测到的人数: {personCount} 人
        </div>
        <p>
          基于 <code>tensorflow.js</code> 的浏览器端实时人数检测系统
        </p>
        <p>
          当前模型: <code className="code">{modelName}</code>
        </p>
        <div className="model-switcher">
          <button 
            onClick={() => switchModel("yolov8n")} 
            className={modelName === "yolov8n" ? "active" : ""}
            disabled={loading.loading}
            style={{color: modelName === "yolov8n" ? "white" : undefined}}
          >
            使用 yolov8n 模型
          </button>
          <button 
            onClick={() => switchModel("best")} 
            className={modelName === "best" ? "active" : ""}
            disabled={loading.loading}
            style={{color: modelName === "best" ? "white" : undefined}}
          >
            使用人体检测模型
          </button>
          <button
            onClick={() => switchModel(obb_model)}
            className={modelName === obb_model ? "active" : ""}
            disabled={loading.loading}
            style={{color: modelName === "obb_model" ? "white" : undefined}}
          >
            使用 {obb_model} 模型
          </button>
           <div className="model-switcher" style={{ marginTop: '15px' }}>
          <button
            onClick={() => switchModel("yolo11n-obb")}
            className={modelName === "yolo11n-obb" ? "active" : ""}
            disabled={loading.loading}
            style={{
              backgroundColor: '#3498db',
              color: modelName === "yolo11n-obb" ? "white" : undefined,
              padding: '8px 15px',
              borderRadius: '4px',
              border: 'none',
              cursor: loading.loading ? 'not-allowed' : 'pointer'
            }}
          >
            使用 yolo11n-obb 模型
          </button>
        </div>
          <button
            onClick={handleRedetect}
            disabled={loading.loading || !model.net}
            style={{
              backgroundColor: '#e74c3c',
              color: 'white',
              marginLeft: '10px',
              padding: '8px 15px',
              cursor: loading.loading || !model.net ? 'not-allowed' : 'pointer'
            }}
          >
            重新检测
          </button>
        </div>

        {/* 添加阈值控制滑动条 */}
        <div className="threshold-controls" style={{ margin: '20px 0', padding: '15px', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
          <div style={{ marginBottom: '15px' }}>
            <label htmlFor="conf-threshold" style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
              置信度阈值: {confThreshold.toFixed(2)}
            </label>
            <input
              id="conf-threshold"
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={confThreshold}
              onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
              style={{ width: '100%', maxWidth: '300px' }}
            />
            <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
            </div>
          </div>

          {modelName.includes("obb") && (
            <div>
              <label htmlFor="nms-threshold" style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
                NMS阈值: 1e-{nmsExponent} ({nmsThreshold.toExponential()})
              </label>
              <input
                id="nms-threshold"
                type="range"
                min="0"
                max="12"
                step="1"
                value={nmsExponent}
                onChange={(e) => setNmsExponent(parseInt(e.target.value))}
                style={{ width: '100%', maxWidth: '300px' }}
              />
              <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
                控制重叠框的过滤强度，较高的值保留更多重叠框
              </div>
            </div>
          )}
        </div>

       

        <p style={{ marginTop: '20px', fontSize: '16px', color: '#666' }}>
          制作者: 潘世维
        </p>
      </div>

      <div className="content">
        <img
          src="#"
          ref={imageRef}
          onLoad={() => {
            if (model.net) {
              const detectFunc = getDetectFunction(true);
              detectFunc(imageRef.current, model, canvasRef.current);
            }
          }}
        />
        <video
          autoPlay
          muted
          ref={cameraRef}
          onPlay={() => {
            if (model.net) {
              const detectFunc = getDetectFunction(false);
              detectFunc(cameraRef.current, model, canvasRef.current);
            }
          }}
        />
        <video
          autoPlay
          muted
          ref={videoRef}
          onPlay={() => {
            if (model.net) {
              const detectFunc = getDetectFunction(false);
              detectFunc(videoRef.current, model, canvasRef.current);
            }
          }}
        />
        <canvas 
          ref={canvasRef} 
          width={model.inputShape[1] || 640} 
          height={model.inputShape[2] || 640} 
        />
      </div>

      <ButtonHandler imageRef={imageRef} cameraRef={cameraRef} videoRef={videoRef} />
    </div>
  );
};

export default App;
