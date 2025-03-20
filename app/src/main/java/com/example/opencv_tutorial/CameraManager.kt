package com.example.opencv_tutorial

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.media.Image
import android.os.Build
import android.util.Log
import android.util.Size
import android.view.Surface
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.IOException
import kotlin.math.roundToInt
import org.opencv.core.Size as OpenCVSize
import java.util.Locale

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

    // Track the current rotation
    private var currentRotation = 0

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
            camera.cameraControl.enableTorch(true)
            
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
     * Process image from camera with optimized handling
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
        
        // Get the image rotation from camera
        val imageRotation = image.imageInfo.rotationDegrees
        
        // Get device orientation
        val deviceOrientation = getDeviceOrientation()
        
        // Store rotation for coordinate transformation and logging
        currentRotation = imageRotation
        
        // Log rotation values for debugging
        Log.d(TAG, "Processing image: rotation=$imageRotation, device orientation=$deviceOrientation, " +
              "dimensions=${image.width}x${image.height}, facing=${if (lensFacing == CameraSelector.LENS_FACING_FRONT) "FRONT" else "BACK"}")

        // Convert image to bitmap in IO dispatcher
        processingScope.launch {
            try {
                // Convert image to bitmap (efficient conversion)
                val bitmap = image.toBitmap()
                
                // PRE-PROCESSING: Prepare the bitmap for inference by ensuring correct orientation for the model
                val preparedBitmap = prepareForInference(bitmap, imageRotation, deviceOrientation)
                
                // Log prepared bitmap dimensions
                Log.d(TAG, "Prepared bitmap for inference: ${preparedBitmap.width}x${preparedBitmap.height}")
                
                // Declare display bitmap at outer scope with a default value
                var displayBitmap = bitmap
                
                // Get detector instance from the parent activity
                val detector = YOLODetectorProvider.getDetector(context)
                
                if (detector != null) {
                    // Run detection on the properly oriented bitmap
                    val detections = detector.detect(preparedBitmap)
                    
                    // POST-PROCESSING: Prepare the bitmap for display and correct detection coordinates
                    displayBitmap = prepareForDisplay(bitmap, imageRotation, deviceOrientation)
                    
                    // Transform detection coordinates to match the display bitmap
                    val correctedDetections = transformDetectionsForDisplay(
                        detections, 
                        preparedBitmap.width, preparedBitmap.height,
                        displayBitmap.width, displayBitmap.height,
                        imageRotation,
                        deviceOrientation
                    )
                    
                    // Track performance
                    val processingTime = System.currentTimeMillis() - startTime
                    performanceTracker.addMeasurement(processingTime)
                    
                    // Log performance periodically
                    if (performanceTracker.totalFrames % 30 == 0) {
                        val avgFps = 1000.0 / performanceTracker.getAverageProcessingTime()
                        Log.d(TAG, "Avg processing time: ${performanceTracker.getAverageProcessingTime()}ms, " +
                                "FPS: ${"%.1f".format(avgFps)}, " +
                                "Detections: ${correctedDetections.size}")
                    }
                    
                    // Notify UI on main thread with display bitmap and corrected detections
                    withContext(Dispatchers.Main) {
                        onFrameProcessed?.invoke(displayBitmap, correctedDetections, processingTime)
                    }
                    
                    // Clean up if needed
                    if (preparedBitmap != bitmap && preparedBitmap != displayBitmap) {
                        preparedBitmap.recycle()
                    }
                }
                
                // Clean up bitmap if needed and not reused
                if (bitmap != preparedBitmap && bitmap != displayBitmap && bitmap != viewFinder.bitmap) {
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
     * Get the current device orientation
     */
    private fun getDeviceOrientation(): Int {
        val display = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            context.display
        } else {
            @Suppress("DEPRECATION")
            (context as? AppCompatActivity)?.windowManager?.defaultDisplay
        }
        
        return when (display?.rotation) {
            Surface.ROTATION_0 -> 0      // Portrait
            Surface.ROTATION_90 -> 90    // Landscape right
            Surface.ROTATION_180 -> 180  // Portrait upside down
            Surface.ROTATION_270 -> 270  // Landscape left
            else -> 0  // Default to portrait if unknown
        }
    }

    /**
     * Prepare bitmap for model inference with proper orientation
     * Ensures the bitmap is in the orientation expected by the model
     */
    private fun prepareForInference(bitmap: Bitmap, imageRotation: Int, deviceOrientation: Int): Bitmap {
        // For YOLO, we typically want the image in its natural orientation (not rotated)
        // The model expects images in natural orientation, so we need to correct camera rotation
        
        // Most YOLO models expect the image in the format they were trained on
        // which is typically natural orientation (like viewing an image in a photo viewer)
        
        try {
            // Create transformation matrix
            val matrix = Matrix()
            
            // For back camera:
            // - In portrait mode: rotate 90 degrees (phone is upright, camera sensor is landscape)
            // - In landscape mode: no rotation (both phone and camera sensor are landscape)
            
            // For front camera:
            // - In portrait mode: rotate 90 degrees and mirror horizontally
            // - In landscape mode: mirror horizontally (selfie mirroring)
            
            val isFrontCamera = lensFacing == CameraSelector.LENS_FACING_FRONT
            val isPortrait = deviceOrientation == 0 || deviceOrientation == 180
            
            if (isPortrait) {
                // Portrait mode - camera sensor is rotated 90 degrees from natural orientation
                matrix.postRotate(90f)
                
                // Mirror for front camera
                if (isFrontCamera) {
                    matrix.postScale(-1f, 1f)
                }
            } else {
                // Landscape mode - camera sensor is aligned with natural orientation
                // but may need mirroring for front camera
                if (isFrontCamera) {
                    matrix.postScale(-1f, 1f)
                }
            }
            
            // Create transformed bitmap
            val preparedBitmap = Bitmap.createBitmap(
                bitmap,
                0, 0,
                bitmap.width, bitmap.height,
                matrix,
                true
            )
            
            Log.d(TAG, "Prepared inference bitmap: orientation=${if (isPortrait) "portrait" else "landscape"}, " +
                      "front=${isFrontCamera}, rotation=${imageRotation}")
            
            return preparedBitmap
        } catch (e: Exception) {
            Log.e(TAG, "Error preparing bitmap for inference: ${e.message}")
            return bitmap
        }
    }

    /**
     * Prepare bitmap for display with proper orientation
     * Ensures the bitmap is oriented correctly for the screen
     */
    private fun prepareForDisplay(bitmap: Bitmap, imageRotation: Int, deviceOrientation: Int): Bitmap {
        try {
            // Create transformation matrix
            val matrix = Matrix()
            
            // The display orientation needs to match the device orientation
            val isFrontCamera = lensFacing == CameraSelector.LENS_FACING_FRONT
            val isPortrait = deviceOrientation == 0 || deviceOrientation == 180
            
            if (isPortrait) {
                // Portrait orientation - rotate camera output to match device orientation
                // Camera sensor is landscape, device is portrait, so rotate 90 degrees
                matrix.postRotate(90f)
                
                // Additional rotation for different device orientations
                if (deviceOrientation == 180) {
                    matrix.postRotate(180f)
                }
                
                // Mirror for front camera
                if (isFrontCamera) {
                    matrix.postScale(-1f, 1f) 
                }
            } else {
                // Landscape orientation
                // Different handling based on landscape left vs right
                if (deviceOrientation == 270) {
                    matrix.postRotate(180f)
                }
                
                // Mirror for front camera
                if (isFrontCamera) {
                    matrix.postScale(-1f, 1f)
                }
            }
            
            // Create transformed bitmap
            val displayBitmap = Bitmap.createBitmap(
                bitmap,
                0, 0,
                bitmap.width, bitmap.height,
                matrix,
                true
            )
            
            Log.d(TAG, "Prepared display bitmap: orientation=${if (isPortrait) "portrait" else "landscape"}, " +
                      "front=${isFrontCamera}, deviceOrientation=${deviceOrientation}")
            
            return displayBitmap
        } catch (e: Exception) {
            Log.e(TAG, "Error preparing bitmap for display: ${e.message}")
            return bitmap
        }
    }

    /**
     * Transform detection coordinates to match the display bitmap
     */
    private fun transformDetectionsForDisplay(
        detections: List<YOLO11Detector.Detection>,
        inferenceWidth: Int, inferenceHeight: Int,
        displayWidth: Int, displayHeight: Int,
        imageRotation: Int,
        deviceOrientation: Int
    ): List<YOLO11Detector.Detection> {
        // Log the transformation parameters
        Log.d(TAG, "Transforming detections: inference=${inferenceWidth}x${inferenceHeight}, " +
                  "display=${displayWidth}x${displayHeight}, " +
                  "imageRotation=${imageRotation}, deviceOrientation=${deviceOrientation}")
        
        // Calculate scale factors if inference and display bitmaps have different dimensions
        val scaleX = displayWidth.toFloat() / inferenceWidth
        val scaleY = displayHeight.toFloat() / inferenceHeight
        
        val isFrontCamera = lensFacing == CameraSelector.LENS_FACING_FRONT
        val isPortrait = deviceOrientation == 0 || deviceOrientation == 180
        
        return detections.map { detection ->
            // Start with the original coordinates (normalized to the inference bitmap)
            val originalX = detection.box.x
            val originalY = detection.box.y
            val originalWidth = detection.box.width
            val originalHeight = detection.box.height
            
            // Variables to hold the transformed coordinates
            var newX: Float
            var newY: Float
            var newWidth: Float
            var newHeight: Float
            
            if (isPortrait) {
                // In portrait mode, width and height are swapped due to 90-degree rotation
                // X becomes Y, Y becomes (width - X - W)
                if (isFrontCamera && (deviceOrientation == 0 || deviceOrientation == 180)) {
                    // Front camera in portrait mode (mirrored)
                    newX = originalY * scaleX
                    newY = (inferenceWidth - originalX - originalWidth) * scaleY
                    newWidth = originalHeight * scaleX
                    newHeight = originalWidth * scaleY
                    
                    // Additional flip for upside-down portrait
                    if (deviceOrientation == 180) {
                        newY = displayHeight - newY - newHeight
                    }
                } else {
                    // Back camera in portrait mode
                    newX = originalY * scaleX
                    newY = (inferenceWidth - originalX - originalWidth) * scaleY
                    newWidth = originalHeight * scaleX
                    newHeight = originalWidth * scaleY
                    
                    // Additional handling for upside-down portrait
                    if (deviceOrientation == 180) {
                        newY = displayHeight - newY - newHeight
                    }
                }
            } else {
                // In landscape mode
                if (isFrontCamera) {
                    // Front camera in landscape mode (mirrored horizontally)
                    newX = (inferenceWidth - originalX - originalWidth) * scaleX
                    newY = originalY * scaleY
                    newWidth = originalWidth * scaleX
                    newHeight = originalHeight * scaleY
                    
                    // Additional flip for landscape left
                    if (deviceOrientation == 270) {
                        newY = displayHeight - newY - newHeight
                        newX = displayWidth - newX - newWidth
                    }
                } else {
                    // Back camera in landscape mode
                    newX = originalX * scaleX
                    newY = originalY * scaleY
                    newWidth = originalWidth * scaleX
                    newHeight = originalHeight * scaleY
                    
                    // Additional flip for landscape left
                    if (deviceOrientation == 270) {
                        newY = displayHeight - newY - newHeight
                        newX = displayWidth - newX - newWidth
                    }
                }
            }
            
            // Create a new detection with the transformed coordinates
            YOLO11Detector.Detection(
                classId = detection.classId,
                conf = detection.conf,
                box = YOLO11Detector.BoundingBox(
                    x = newX.toInt(),
                    y = newY.toInt(),
                    width = newWidth.toInt(),
                    height = newHeight.toInt()
                )
            )
        }
    }

    /**
     * Extension function to convert ImageProxy to Bitmap
     */
    private suspend fun ImageProxy.toBitmap(): Bitmap = withContext(Dispatchers.IO) {
        try {
            // For YUV_420_888 format (most common in CameraX), use the more reliable method
            if (format == ImageFormat.YUV_420_888) {
                return@withContext convertYuv420ToBitmap()
            }
            
            // For JPEG format, use a different approach
            if (format == ImageFormat.JPEG) {
                val buffer = planes[0].buffer
                val bytes = ByteArray(buffer.capacity())
                buffer.rewind()
                buffer.get(bytes)
                return@withContext BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            }

            // For other formats, try a safer approach that doesn't rely on copyPixelsFromBuffer
            val yuvImage = when (format) {
                ImageFormat.YUV_420_888 -> convertYuv420ToBitmap()
                else -> {
                    // Create a new bitmap and draw directly from source
                    val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                    val canvas = android.graphics.Canvas(bitmap)
                    val paint = android.graphics.Paint()
                    paint.color = android.graphics.Color.BLACK
                    canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), paint)
                    
                    // Try to draw something useful if possible
                    try {
                        val mat = Mat(height, width, CvType.CV_8UC4)
                        val rgba = Mat()
                        Imgproc.cvtColor(mat, rgba, Imgproc.COLOR_BGR2RGBA)
                        Utils.matToBitmap(rgba, bitmap)
                        mat.release()
                        rgba.release()
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to convert to bitmap: ${e.message}")
                    }
                    
                    bitmap
                }
            }
            
            return@withContext yuvImage
            
        } catch (e: Exception) {
            // Final fallback: create an emergency bitmap
            Log.e(TAG, "All bitmap conversion methods failed: ${e.message}")
            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val canvas = android.graphics.Canvas(bitmap)
            val paint = android.graphics.Paint()
            paint.color = android.graphics.Color.RED
            canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), paint)
            return@withContext bitmap
        }
    }

    /**
     * Convert YUV_420_888 format ImageProxy to Bitmap
     * Comprehensive implementation that handles different pixel strides and formats
     */
    private suspend fun ImageProxy.convertYuv420ToBitmap(): Bitmap = withContext(Dispatchers.IO) {
        // Get image planes
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]
        
        // Get plane buffers
        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        
        // Get buffer sizes
        val ySize = yBuffer.remaining()
        
        // Create properly sized byte array for NV21 format
        val nv21Size = width * height * 3 / 2  // This is the standard size for YUV420/NV21 format
        val nv21 = ByteArray(nv21Size)
        
        // Get strides for proper conversion
        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        
        // Copy Y plane with handling for different strides
        var position = 0
        
        // Reset buffer position
        yBuffer.rewind()
        
        if (yRowStride == width && yPixelStride == 1) {
            // Fast path - can copy in one go
            yBuffer.get(nv21, 0, width * height)
            position = width * height
        } else {
            // Slower path - copy row by row
            for (row in 0 until height) {
                // Position buffer at start of row
                yBuffer.position(row * yRowStride)
                
                // Copy pixels from this row
                for (col in 0 until width) {
                    nv21[position++] = yBuffer.get()
                    // Skip extra pixels in row if needed
                    if (col < width - 1 && yPixelStride > 1) {
                        yBuffer.position(yBuffer.position() + yPixelStride - 1)
                    }
                }
            }
        }
        
        // Calculate U/V positioning
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride
        val uvWidth = width / 2
        val uvHeight = height / 2
        
        // Copy U/V data
        for (row in 0 until uvHeight) {
            for (col in 0 until uvWidth) {
                // Calculate buffer positions
                val uvRow = row
                val uvCol = col
                val uvIndex = uvRow * uvRowStride + uvCol * uvPixelStride
                
                // Add V value then U value for NV21 format
                if (uvIndex < vBuffer.capacity())
                    nv21[position++] = vBuffer.get(uvIndex)
                else
                    nv21[position++] = 0  // Default value if out of bounds
                    
                if (uvIndex < uBuffer.capacity())    
                    nv21[position++] = uBuffer.get(uvIndex)
                else
                    nv21[position++] = 0  // Default value if out of bounds
            }
        }
        
        // Convert to Bitmap via YuvImage -> JPEG -> Bitmap
        val yuvImage = android.graphics.YuvImage(
            nv21, 
            android.graphics.ImageFormat.NV21,
            width, height, null
        )
        
        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 100, out)
        val imageBytes = out.toByteArray()
        
        return@withContext BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    /**
     * Convert ImageProxy to YUV Mat with proper handling of stride and formats
     */
    private fun imageToYuvMat(image: ImageProxy): Mat {
        val width = image.width
        val height = image.height
        val planes = image.planes
        
        // Calculate needed buffer size based on plane strides
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]
        
        // Allocate a single continuous buffer for all YUV data
        val yuvTotalSize = width * height * 3 / 2 // Standard size for YUV420
        val yuvMat = Mat(yuvTotalSize, 1, CvType.CV_8UC1)
        val yuvBytes = ByteArray(yuvTotalSize)
        
        // Get buffers
        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        
        // Get strides
        val yPixelStride = yPlane.pixelStride
        val yRowStride = yPlane.rowStride
        val uPixelStride = uPlane.pixelStride 
        val uRowStride = uPlane.rowStride
        val vPixelStride = vPlane.pixelStride
        val vRowStride = vPlane.rowStride
        
        // If pixel stride is 1, we can copy rows directly for Y plane
        var position = 0
        if (yPixelStride == 1) {
            for (row in 0 until height) {
                yBuffer.position(row * yRowStride)
                yBuffer.get(yuvBytes, position, width)
                position += width
            }
        } else {
            // Handle y plane with pixel stride > 1
            for (row in 0 until height) {
                yBuffer.position(row * yRowStride)
                for (col in 0 until width) {
                    yuvBytes[position++] = yBuffer.get()
                    if (col < width - 1) yBuffer.position(yBuffer.position() + yPixelStride - 1)
                }
            }
        }
        
        // Process UV planes - correctly interleave U and V planes
        val chromaHeight = height / 2
        val chromaWidth = width / 2
        
        // Check if UV channels are interleaved
        val isUVInterleaved = uPixelStride == 2 && vPixelStride == 2 && 
                             uPlane.buffer.capacity() == vPlane.buffer.capacity()
        val isVUFormat = isUVInterleaved && uPlane.buffer.compareTo(vPlane.buffer) > 0
        
        if (isUVInterleaved) {
            // Handle interleaved UV/VU format
            for (row in 0 until chromaHeight) {
                for (col in 0 until chromaWidth) {
                    val uvIndex = (row * uRowStride) + (col * 2)
                    if (isVUFormat) {
                        // NV21: VU order
                        yuvBytes[position++] = vBuffer.get(uvIndex)  // V first for NV21
                        yuvBytes[position++] = uBuffer.get(uvIndex)  // U second
                    } else {
                        // NV12: UV order
                        yuvBytes[position++] = uBuffer.get(uvIndex)  // U
                        yuvBytes[position++] = vBuffer.get(uvIndex)  // V
                    }
                }
            }
        } else {
            // Fall back to pixel-by-pixel copy for non-interleaved formats
            for (row in 0 until chromaHeight) {
                for (col in 0 until chromaWidth) {
                    val uIndex = (row * uRowStride) + (col * uPixelStride)
                    val vIndex = (row * vRowStride) + (col * vPixelStride)
                    // Default to NV21 format
                    yuvBytes[position++] = vBuffer.get(vIndex)  // V
                    yuvBytes[position++] = uBuffer.get(uIndex)  // U
                }
            }
        }
        
        // Put the data into the Mat
        yuvMat.put(0, 0, yuvBytes)
        
        // Reshape to proper dimensions for OpenCV processing
        val yuv420sp = yuvMat.reshape(1, height + height/2)
        return yuv420sp
    }

    /**
     * Enhanced transformation of detection coordinates based on orientation
     */
    private fun transformDetectionsBasedOnOrientation(
        detections: List<YOLO11Detector.Detection>,
        width: Int,
        height: Int,
        rotation: Int
    ): List<YOLO11Detector.Detection> {
        // Log input parameters for debugging
        Log.d(TAG, "Transforming detections: width=$width, height=$height, rotation=$rotation")
        Log.d(TAG, "Camera facing: ${if (lensFacing == CameraSelector.LENS_FACING_FRONT) "FRONT" else "BACK"}")
        
        // Determine if we need to flip coordinates for front camera
        val isFrontCamera = lensFacing == CameraSelector.LENS_FACING_FRONT
        
        return detections.map { detection ->
            // Log original detection coordinates
            Log.d(TAG, "Original detection: x=${detection.box.x}, y=${detection.box.y}, " +
                     "w=${detection.box.width}, h=${detection.box.height}")
            
            val correctedBox = when (rotation) {
                0 -> {
                    // No rotation needed
                    detection.box
                }
                90 -> {
                    // For 90-degree rotation:
                    // Map (x,y,w,h) => (y, width-x-w, h, w)
                    YOLO11Detector.BoundingBox(
                        x = detection.box.y,
                        y = width - detection.box.x - detection.box.width,
                        width = detection.box.height,
                        height = detection.box.width
                    )
                }
                180 -> {
                    // For 180-degree rotation:
                    // Map (x,y,w,h) => (width-x-w, height-y-h, w, h)
                    YOLO11Detector.BoundingBox(
                        x = width - detection.box.x - detection.box.width,
                        y = height - detection.box.y - detection.box.height,
                        width = detection.box.width,
                        height = detection.box.height
                    )
                }
                270 -> {
                    // For 270-degree rotation:
                    // Map (x,y,w,h) => (height-y-h, x, h, w)
                    YOLO11Detector.BoundingBox(
                        x = height - detection.box.y - detection.box.height,
                        y = detection.box.x,
                        width = detection.box.height,
                        height = detection.box.width
                    )
                }
                else -> {
                    // Unexpected rotation value, log and use original
                    Log.e(TAG, "Unexpected rotation value: $rotation, using original coordinates")
                    detection.box
                }
            }
            
            // Apply front camera mirroring if needed
            val finalBox = if (isFrontCamera && (rotation == 90 || rotation == 270)) {
                // For front camera in portrait orientation, need to mirror horizontally
                YOLO11Detector.BoundingBox(
                    x = width - correctedBox.x - correctedBox.width,
                    y = correctedBox.y,
                    width = correctedBox.width,
                    height = correctedBox.height
                )
            } else {
                correctedBox
            }
            
            // Log transformed coordinates
            Log.d(TAG, "Transformed detection: x=${finalBox.x}, y=${finalBox.y}, " +
                     "w=${finalBox.width}, h=${finalBox.height}")
            
            // Create a new detection with the transformed coordinates
            YOLO11Detector.Detection(
                classId = detection.classId,
                conf = detection.conf,
                box = finalBox
            )
        }
    }

    /**
     * Improved bitmap rotation with proper orientation handling
     */
    private fun rotateBitmapIfNeeded(bitmap: Bitmap, rotation: Int): Bitmap {
        // Skip if no rotation needed
        if (rotation == 0) return bitmap
        
        try {
            // Get device rotation from activity context
            val display = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                context.display
            } else {
                @Suppress("DEPRECATION")
                (context as? AppCompatActivity)?.windowManager?.defaultDisplay
            }
            
            // Default to 0 (portrait) if display info can't be determined
            val deviceOrientation = when (display?.rotation) {
                Surface.ROTATION_0 -> 0    // Portrait
                Surface.ROTATION_90 -> 90  // Landscape right
                Surface.ROTATION_180 -> 180 // Portrait upside down
                Surface.ROTATION_270 -> 270 // Landscape left
                else -> 0
            }
            
            // Log debug information
            Log.d(TAG, "Device orientation: $deviceOrientation, Camera rotation: $rotation, " +
                     "Camera facing: ${if (lensFacing == CameraSelector.LENS_FACING_FRONT) "FRONT" else "BACK"}")
            
            // Create transformation matrix
            val matrix = Matrix()
            
            // Determine if we're in portrait or landscape mode
            val isPortrait = deviceOrientation == 0 || deviceOrientation == 180
            val isLandscape = deviceOrientation == 90 || deviceOrientation == 270
            
            // Handle different rotation scenarios based on device orientation and camera facing
            if (lensFacing == CameraSelector.LENS_FACING_BACK) {
                // Back camera rotation handling
                if (isPortrait) {
                    // Portrait mode - rotate 90 degrees counterclockwise
                    matrix.postRotate(rotation.toFloat())
                } else {
                    // Landscape mode - align camera output with display orientation
                    val correctedRotation = (rotation + deviceOrientation) % 360
                    matrix.postRotate(correctedRotation.toFloat())
                }
            } else {
                // Front camera rotation handling - needs mirroring
                if (isPortrait) {
                    // Front camera in portrait mode - rotate and mirror
                    matrix.postScale(-1f, 1f) // Mirror horizontally
                    matrix.postRotate(rotation.toFloat())
                } else {
                    // Front camera in landscape mode
                    val correctedRotation = (rotation + deviceOrientation) % 360
                    matrix.postScale(-1f, 1f) // Mirror horizontally
                    matrix.postRotate(correctedRotation.toFloat())
                }
            }
            
            // Log transformation details
            Log.d(TAG, "Applied transformation: rotation=$rotation, mirrored=${lensFacing == CameraSelector.LENS_FACING_FRONT}")
            
            // Create rotated bitmap
            val rotatedBitmap = Bitmap.createBitmap(
                bitmap, 
                0, 0, 
                bitmap.width, bitmap.height, 
                matrix, 
                true
            )
            
            // Performance optimization: only recycle original if it's different from what we started with
            if (bitmap != rotatedBitmap && bitmap != viewFinder.bitmap) {
                bitmap.recycle()
            }
            
            // Log output dimensions for verification
            Log.d(TAG, "Rotated bitmap dimensions: ${rotatedBitmap.width}x${rotatedBitmap.height}")
            
            return rotatedBitmap
        } catch (e: Exception) {
            Log.e(TAG, "Error during bitmap rotation: ${e.message}")
            return bitmap
        }
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
     * Enhanced orientation change handler
     */
    fun handleOrientationChange(orientation: Int) {
        try {
            // Log the orientation change
            Log.d(TAG, "Orientation changed to: $orientation")
            
            // Get the display rotation
            val display = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                context.display
            } else {
                @Suppress("DEPRECATION")
                (context as? android.app.Activity)?.windowManager?.defaultDisplay
            }
            
            val rotation = when (display?.rotation) {
                android.view.Surface.ROTATION_0 -> 0
                android.view.Surface.ROTATION_90 -> 90
                android.view.Surface.ROTATION_180 -> 180
                android.view.Surface.ROTATION_270 -> 270
                else -> 0
            }
            
            // Log the detected display rotation
            Log.d(TAG, "Display rotation: $rotation")
            
            // Store current rotation for later use in transformations
            currentRotation = rotation
            
            // No need to rebind camera use cases - CameraX handles display rotation automatically
        } catch (e: Exception) {
            Log.e(TAG, "Error handling orientation change: ${e.message}", e)
        }
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
    private const val TAG = "YOLODetectorProvider"
    
    fun getDetector(context: Context): YOLO11Detector? {
        if (detector == null) {
            synchronized(initLock) {
                if (detector == null) {
                    try {
                        // Try multiple model variants, matching MainActivity's approach
                        val modelVariants = listOf(
                            "best_float16.tflite",  // Try float16 first (smaller, works on many devices)
                            "best_float32.tflite",  // Try float32 as fallback (more compatible but larger)
                            "best.tflite"           // Try default naming as last resort
                        )
                        
                        val labelsPath = "classes.txt"
                        
                        // Check GPU compatibility
                        val useGPU = checkGpuCompatibility(context)
                        Log.d(TAG, "GPU acceleration decision: $useGPU")
                        
                        // Try model variants in sequence until one works
                        var lastException: Exception? = null
                        
                        for (modelFile in modelVariants) {
                            try {
                                // Check if file exists in assets
                                try {
                                    context.assets.open(modelFile).close()
                                } catch (e: IOException) {
                                    Log.d(TAG, "Model file $modelFile not found in assets, skipping")
                                    continue
                                }
                                
                                Log.d(TAG, "Attempting to load model: $modelFile")
                                
                                // Create detector with current model variant
                                detector = YOLO11Detector(
                                    context = context,
                                    modelPath = modelFile,
                                    labelsPath = labelsPath,
                                    useGPU = useGPU
                                )
                                
                                // If we get here, initialization succeeded
                                Log.d(TAG, "Successfully initialized detector with model: $modelFile")
                                break
                            } catch (e: Exception) {
                                Log.e(TAG, "Failed to initialize with model $modelFile: ${e.message}")
                                lastException = e
                                
                                // If this is GPU mode and failed, try again with CPU
                                if (useGPU) {
                                    try {
                                        Log.d(TAG, "Retrying model $modelFile with CPU only")
                                        detector = YOLO11Detector(
                                            context = context,
                                            modelPath = modelFile,
                                            labelsPath = labelsPath,
                                            useGPU = false
                                        )
                                        
                                        // If we get here, CPU initialization succeeded
                                        Log.d(TAG, "Successfully initialized detector with CPU and model: $modelFile")
                                        break
                                    } catch (cpuEx: Exception) {
                                        Log.e(TAG, "CPU fallback also failed for $modelFile: ${cpuEx.message}")
                                    }
                                }
                            }
                        }
                        
                        // If no model worked, log error
                        if (detector == null && lastException != null) {
                            Log.e(TAG, "Failed to initialize detector with any available model", lastException)
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to initialize detector: ${e.message}")
                        return null
                    }
                }
            }
        }
        return detector
    }

    /**
     * Check if the device is compatible with GPU acceleration
     */
    private fun checkGpuCompatibility(context: Context): Boolean {
        // Check if GPU delegation is supported
        val compatList = org.tensorflow.lite.gpu.CompatibilityList()
        val isGpuSupported = compatList.isDelegateSupportedOnThisDevice
        
        // Check if running on emulator
        val isEmulator = android.os.Build.FINGERPRINT.contains("generic") ||
                android.os.Build.FINGERPRINT.startsWith("unknown") ||
                android.os.Build.MODEL.contains("google_sdk") ||
                android.os.Build.MODEL.contains("Emulator") ||
                android.os.Build.MODEL.contains("Android SDK")
        
        // Check known problematic device models and manufacturers
        val deviceModel = android.os.Build.MODEL.toLowerCase(java.util.Locale.ROOT)
        val manufacturer = android.os.Build.MANUFACTURER.toLowerCase(java.util.Locale.ROOT)
        
        // List of known problematic device patterns
        val problematicPatterns = listOf(
            "mali-g57", "mali-g72", "mali-g52", "mali-g76",  // Some Mali GPUs have TFLite issues
            "adreno 6", "adreno 5",                          // Some older Adreno GPUs
            "mediatek", "mt6", "helio"                        // Some MediaTek chips
        )
        
        val isProblematicDevice = problematicPatterns.any { pattern ->
            deviceModel.contains(pattern) || manufacturer.contains(pattern)
        }
        
        // Check Android version - some versions have known TFLite GPU issues
        val androidVersion = android.os.Build.VERSION.SDK_INT
        val isProblematicAndroidVersion = androidVersion < android.os.Build.VERSION_CODES.P  // Android 9-
        
        // Check available memory - GPU acceleration needs sufficient memory
        val memoryInfo = android.app.ActivityManager.MemoryInfo()
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        activityManager.getMemoryInfo(memoryInfo)
        
        val availableMem = memoryInfo.availMem / (1024 * 1024)  // Convert to MB
        val lowMemory = availableMem < 200  // Less than 200MB available
        
        // Final decision based on all factors
        return isGpuSupported &&
                !isEmulator &&
                !isProblematicDevice &&
                !isProblematicAndroidVersion &&
                !lowMemory
    }
    
    fun releaseDetector() {
        detector?.close()
        detector = null
    }
}
