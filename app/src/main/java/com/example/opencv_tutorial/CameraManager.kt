package com.example.opencv_tutorial

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import android.util.Size
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import kotlin.math.roundToInt

/**
 * Manages camera preview and frame processing for real-time object detection
 */
class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val viewFinder: PreviewView
) {
    // Executor for camera operations
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    
    // Coroutine scope for processing frames
    private val processingScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    // Tracking for frame processing performance
    private val performanceTracker = PerformanceTracker(10) // Average over 10 frames
    
    // Camera properties
    private var camera: Camera? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var lensFacing = CameraSelector.LENS_FACING_BACK
    
    // Resolution management
    private val preferredResolution = Size(640, 640) // Match model input size for efficiency
    
    // Processing state flags
    @Volatile private var isProcessing = false
    @Volatile private var processingSkipCount = 0
    
    // Frame processing callback
    var onFrameProcessed: ((Bitmap, List<YOLO11Detector.Detection>, Long) -> Unit)? = null
    
    companion object {
        private const val TAG = "CameraManager"
        private const val MAX_SKIP_FRAMES = 2 // Skip at most 2 frames when busy
    }

    /**
     * Start camera preview and analysis
     */
    fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        
        cameraProviderFuture.addListener({
            try {
                val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
                bindCameraUseCases(cameraProvider)
                Log.d(TAG, "Camera initialization completed")
            } catch (e: Exception) {
                Log.e(TAG, "Camera initialization failed: ${e.message}", e)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    /**
     * Bind camera use cases
     */
    private fun bindCameraUseCases(cameraProvider: ProcessCameraProvider) {
        // Select camera based on lens facing
        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(lensFacing)
            .build()

        // Preview use case
        val preview = Preview.Builder()
            .setTargetResolution(preferredResolution)
            .build()
            .also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }

        // Image analysis use case
        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetResolution(preferredResolution)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(cameraExecutor) { image ->
                    processImage(image)
                }
            }

        try {
            // Unbind any bound use cases before rebinding
            cameraProvider.unbindAll()

            // Bind use cases to camera
            camera = cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageAnalyzer
            )
            
            configureCamera()
            
        } catch (e: Exception) {
            Log.e(TAG, "Use case binding failed", e)
        }
    }
    
    /**
     * Configure camera settings for optimal detection
     */
    private fun configureCamera() {
        camera?.let { camera ->
            // Enable auto-exposure, auto-focus
            camera.cameraControl.enableTorch(false)
            
            try {
                // Set optimal camera mode for video/preview
                val cameraControl = camera.cameraControl
                val cameraInfo = camera.cameraInfo
                
                // Try to set frame rate range for smoother preview
                if (BuildConfig.DEBUG) {
                    // Log exposure state information instead of checking hasCamera
                    Log.d(TAG, "Camera exposure state: ${cameraInfo.exposureState}")
                }
                
                // Request auto-focus mode for moving objects
                cameraControl.cancelFocusAndMetering()
            } catch (e: Exception) {
                Log.e(TAG, "Error configuring camera: ${e.message}")
            }
        }
    }

    /**
     * Process image from camera
     */
    private fun processImage(image: ImageProxy) {
        // Skip processing if we're still working on the previous frame
        if (isProcessing) {
            processingSkipCount++
            if (processingSkipCount <= MAX_SKIP_FRAMES) {
                image.close()
                return
            }
            // Force processing after skipping too many frames
            processingSkipCount = 0
        }

        isProcessing = true
        val startTime = System.currentTimeMillis()

        // Convert image to bitmap in IO dispatcher
        processingScope.launch {
            try {
                // Get bitmap from image proxy (efficient conversion)
                val bitmap = image.toBitmap()
                
                // Reorient bitmap if needed
                val rotatedBitmap = rotateBitmapIfNeeded(bitmap, image.imageInfo.rotationDegrees)
                
                // Get detector instance from the parent activity
                val detector = YOLODetectorProvider.getDetector(context)
                
                if (detector != null) {
                    // Run detection
                    val detections = detector.detect(rotatedBitmap)
                    
                    // Track performance
                    val processingTime = System.currentTimeMillis() - startTime
                    performanceTracker.addMeasurement(processingTime)
                    
                    // Log performance every 30 frames
                    if (performanceTracker.totalFrames % 30 == 0) {
                        val avgFps = 1000.0 / performanceTracker.getAverageProcessingTime()
                        Log.d(TAG, "Avg processing time: ${performanceTracker.getAverageProcessingTime()}ms, " +
                                "FPS: ${String.format("%.1f", avgFps)}, " +
                                "Detections: ${detections.size}")
                    }
                    
                    // Notify UI on main thread
                    withContext(Dispatchers.Main) {
                        onFrameProcessed?.invoke(rotatedBitmap, detections, processingTime)
                    }
                }
                
                // Clean up temporary bitmaps if needed
                if (rotatedBitmap !== bitmap) {
                    bitmap.recycle()
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Error processing frame: ${e.message}", e)
            } finally {
                // Always close the image and reset processing state when done
                image.close()
                isProcessing = false
            }
        }
    }
    
    /**
     * Efficiently convert ImageProxy to Bitmap using OpenCV for better performance
     */
    private suspend fun ImageProxy.toBitmap(): Bitmap = withContext(Dispatchers.Default) {
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer
        
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        
        val nv21 = ByteArray(ySize + uSize + vSize)
        
        // U and V are swapped
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        
        val yuvMat = Mat(height + height / 2, width, org.opencv.core.CvType.CV_8UC1)
        yuvMat.put(0, 0, nv21)
        val rgbMat = Mat()
        
        // Convert YUV to RGB
        Imgproc.cvtColor(yuvMat, rgbMat, Imgproc.COLOR_YUV2RGB_NV21)
        
        // Create bitmap from Mat
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgbMat, bitmap)
        
        // Release Mats
        yuvMat.release()
        rgbMat.release()
        
        bitmap
    }
    
    /**
     * Rotate bitmap if needed based on camera orientation
     */
    private fun rotateBitmapIfNeeded(bitmap: Bitmap, rotation: Int): Bitmap {
        if (rotation == 0) return bitmap
        
        val matrix = Matrix()
        matrix.postRotate(rotation.toFloat())
        
        return Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
        )
    }
    
    /**
     * Switch between front and back camera
     */
    fun switchCamera() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        
        // Get camera provider and rebind use cases
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            bindCameraUseCases(cameraProviderFuture.get())
        }, ContextCompat.getMainExecutor(context))
    }
    
    /**
     * Change detection resolution
     */
    fun setAnalysisResolution(width: Int, height: Int) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                imageAnalyzer?.let {
                    cameraProvider.unbind(it)
                }
                
                imageAnalyzer = ImageAnalysis.Builder()
                    .setTargetResolution(Size(width, height))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { analysis ->
                        analysis.setAnalyzer(cameraExecutor) { image ->
                            processImage(image)
                        }
                    }
                
                // Rebind use cases
                bindCameraUseCases(cameraProvider)
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to update resolution: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(context))
    }
    
    /**
     * Release resources when no longer needed
     */
    fun shutdown() {
        processingScope.cancel()
        cameraExecutor.shutdown()
    }
    
    /**
     * Helper class to track performance metrics
     */
    private class PerformanceTracker(private val windowSize: Int) {
        private val processingTimes = ArrayDeque<Long>(windowSize)
        var totalFrames = 0
            private set
        
        fun addMeasurement(timeMs: Long) {
            if (processingTimes.size >= windowSize) {
                processingTimes.removeFirst()
            }
            processingTimes.add(timeMs)  // Changed from addLast to add
            totalFrames++
        }
        
        fun getAverageProcessingTime(): Double {
            if (processingTimes.isEmpty()) return 0.0
            return processingTimes.average()
        }
    }
}

/**
 * Singleton to provide detector instance to avoid multiple initializations
 */
object YOLODetectorProvider {
    private var detector: YOLO11Detector? = null
    private val initLock = Any()
    
    fun getDetector(context: Context): YOLO11Detector? {
        if (detector == null) {
            synchronized(initLock) {
                if (detector == null) {
                    try {
                        detector = YOLO11Detector(
                            context,
                            "best_float32.tflite", // Model file in assets
                            "coco.txt",       // Labels file in assets
                            useGPU = true     // Enable GPU acceleration
                        )
                    } catch (e: Exception) {
                        Log.e("YOLODetectorProvider", "Failed to initialize detector: ${e.message}")
                        return null
                    }
                }
            }
        }
        return detector
    }
    
    fun releaseDetector() {
        detector?.close()
        detector = null
    }
}
