package com.example.opencv_tutorial

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.Build
import android.os.SystemClock
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round
//import android.util.Log

/**
 * YOLOv11Detector for Android using TFLite and OpenCV
 *
 * This class handles object detection using the YOLOv11 model with TensorFlow Lite
 * for inference and OpenCV for image processing.
 */
class YOLO11Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelsPath: String,
    useGPU: Boolean = true
) {
    // Detection parameters - matching C++ implementation
    companion object {
        // Match the C++ implementation thresholds
        const val CONFIDENCE_THRESHOLD = 0.25f  // Changed from 0.4f to match C++ code
        const val IOU_THRESHOLD = 0.45f         // Changed from 0.3f to match C++ code
        private const val TAG = "YOLO11Detector"
    }

    // Data structures for model and inference
    private var interpreter: Interpreter
    private val classNames: List<String>
    private val classColors: List<IntArray>
    private var gpuDelegate: GpuDelegate? = null

    // Input shape info
    private var inputWidth: Int = 640
    private var inputHeight: Int = 640
    private var isQuantized: Boolean = false
    private var numClasses: Int = 0

    init {
        try {
            // Log starting initialization for debugging purposes
            debug("Initializing YOLO11Detector with model: $modelPath, useGPU: $useGPU")
            debug("Device: ${Build.MANUFACTURER} ${Build.MODEL}, Android ${Build.VERSION.SDK_INT}")

            // Load model with proper options
            val tfliteOptions = Interpreter.Options()

            // GPU Delegate setup with improved validation and error recovery
            if (useGPU) {
                try {
                    val compatList = CompatibilityList()
                    debug("GPU delegate supported on device: ${compatList.isDelegateSupportedOnThisDevice}")

                    if (compatList.isDelegateSupportedOnThisDevice) {
                        // First try to create GPU delegate without configuring options
                        // This can help detect early incompatibilities
                        try {
                            val tempDelegate = GpuDelegate()
                            tempDelegate.close() // Just testing creation
                            debug("Basic GPU delegate creation successful")
                        } catch (e: Exception) {
                            debug("Basic GPU delegate test failed: ${e.message}")
                            throw Exception("Device reports GPU compatible but fails basic delegate test")
                        }

                        debug("Configuring GPU acceleration with safe defaults")

                        // Use conservative GPU delegation options
                        val delegateOptions = GpuDelegate.Options().apply {
                            setPrecisionLossAllowed(true)  // Allow precision loss for better compatibility
                            setQuantizedModelsAllowed(true)  // Allow quantized models
                        }

                        gpuDelegate = GpuDelegate(delegateOptions)
                        tfliteOptions.addDelegate(gpuDelegate)
                        debug("GPU delegate successfully created and added")

                        // Always configure CPU fallback options
                        configureCpuOptions(tfliteOptions)
                    } else {
                        debug("GPU acceleration not supported on this device, using CPU only")
                        configureCpuOptions(tfliteOptions)
                    }
                } catch (e: Exception) {
                    debug("Error setting up GPU acceleration: ${e.message}, stack: ${e.stackTraceToString()}")
                    debug("Falling back to CPU execution")
                    // Clean up any GPU resources
                    try {
                        gpuDelegate?.close()
                    } catch (closeEx: Exception) {
                        debug("Error closing GPU delegate: ${closeEx.message}")
                    }
                    gpuDelegate = null
                    configureCpuOptions(tfliteOptions)
                }
            } else {
                debug("GPU acceleration disabled, using CPU only")
                configureCpuOptions(tfliteOptions)
            }

            // Enhanced model loading with diagnostics
            val modelBuffer: MappedByteBuffer
            try {
                debug("Loading model from assets: $modelPath")
                modelBuffer = loadModelFile(modelPath)
                debug("Model loaded successfully, size: ${modelBuffer.capacity() / 1024} KB")

                // Simple validation - check if buffer size is reasonable
                if (modelBuffer.capacity() < 10000) {
                    throw RuntimeException("Model file appears too small (${modelBuffer.capacity()} bytes)")
                }
            } catch (e: Exception) {
                debug("Failed to load model: ${e.message}")
                throw RuntimeException("Model loading failed: ${e.message}", e)
            }

            // Initialize interpreter with more controlled error handling
            try {
                debug("Creating TFLite interpreter")

                // Add memory management options for large models
                tfliteOptions.setAllowFp16PrecisionForFp32(true) // Reduce memory requirements

                interpreter = Interpreter(modelBuffer, tfliteOptions)
                debug("TFLite interpreter created successfully")

                // Log interpreter details for diagnostics
                val inputTensor = interpreter.getInputTensor(0)
                val inputShape = inputTensor.shape()
                val outputTensor = interpreter.getOutputTensor(0)
                val outputShape = outputTensor.shape()

                debug("Model input shape: ${inputShape.joinToString()}")
                debug("Model output shape: ${outputShape.joinToString()}")
                debug("Input tensor type: ${inputTensor.dataType()}")

                // Capture model input properties
                inputHeight = inputShape[1]
                inputWidth = inputShape[2]
                isQuantized = inputTensor.dataType() == org.tensorflow.lite.DataType.UINT8
                numClasses = outputShape[1] - 4

                debug("Model setup: inputSize=${inputWidth}x${inputHeight}, isQuantized=$isQuantized, numClasses=$numClasses")
            } catch (e: Exception) {
                debug("Failed to initialize interpreter: ${e.message}, stack: ${e.stackTraceToString()}")
                // Clean up resources
                try {
                    gpuDelegate?.close()
                } catch (closeEx: Exception) {
                    debug("Error closing GPU delegate during cleanup: ${closeEx.message}")
                }
                throw RuntimeException("TFLite initialization failed: ${e.message}", e)
            }

            // Load class names
            try {
                classNames = loadClassNames(labelsPath)
                debug("Loaded ${classNames.size} classes from $labelsPath")
                classColors = generateColors(classNames.size)

                if (classNames.size != numClasses) {
                    debug("Warning: Number of classes in label file (${classNames.size}) differs from model output ($numClasses)")
                }
            } catch (e: Exception) {
                debug("Failed to load class names: ${e.message}")
                throw RuntimeException("Failed to load class names", e)
            }

            debug("YOLO11Detector initialization completed successfully")
        } catch (e: Exception) {
            debug("FATAL: Detector initialization failed: ${e.message}")
            debug("Stack trace: ${e.stackTraceToString()}")
            throw e  // Re-throw to ensure caller sees the failure
        }
    }

    /**
     * Configure CPU-specific options for the TFLite interpreter with safer defaults
     */
    private fun configureCpuOptions(options: Interpreter.Options) {
        try {
            // Determine optimal thread count based on device
            val cpuCores = Runtime.getRuntime().availableProcessors()
            // For lower-end devices, use fewer threads to avoid overwhelming the CPU
            val optimalThreads = when {
                cpuCores <= 2 -> 1
                cpuCores <= 4 -> 2
                else -> cpuCores - 2
            }

            options.setNumThreads(optimalThreads)
            options.setUseXNNPACK(true)  // Use XNNPACK for CPU acceleration

            // Add FlatBuffer-related options
            options.setAllowFp16PrecisionForFp32(true)
            options.setAllowBufferHandleOutput(true)

            debug("CPU options configured with $optimalThreads threads")
        } catch (e: Exception) {
            debug("Error configuring CPU options: ${e.message}")
            // Use safe defaults
            options.setNumThreads(1)
        }
    }

    /**
     * Loads the TFLite model file with enhanced error checking
     */
    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        try {
            debug("Loading model from assets: $modelPath")

            // Use TFLiteModelManager for optimized model loading
            val modelManager = TFLiteModelManager(context)

            // First check if model exists and prepare it for device compatibility
            val optimizedModelPath = try {
                modelManager.prepareModelForDevice(modelPath)
            } catch (e: Exception) {
                debug("Failed to optimize model: ${e.message}, falling back to direct asset loading")
                null
            }

            // If optimized model was prepared, load it from file system
            if (optimizedModelPath != null && modelManager.validateModelFile(optimizedModelPath)) {
                debug("Loading optimized model from: $optimizedModelPath")
                return modelManager.loadModelFile(optimizedModelPath)
            }

            // Fallback to direct asset loading if optimization failed
            val assetManager = context.assets

            // Check if file exists
            val assetList = assetManager.list("") ?: emptyArray()
            debug("Available assets: ${assetList.joinToString()}")

            if (!assetList.contains(modelPath)) {
                throw IOException("Model file not found in assets: $modelPath")
            }

            val assetFileDescriptor = assetManager.openFd(modelPath)
            val modelSize = assetFileDescriptor.length
            debug("Model file size: $modelSize bytes")

            // Check if model size is reasonable
            if (modelSize <= 0) {
                throw IOException("Invalid model file size: $modelSize")
            }

            val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = fileInputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength

            debug("Mapping model file: offset=$startOffset, length=$declaredLength")

            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength).also {
                debug("Model buffer capacity: ${it.capacity()} bytes")
            }
        } catch (e: Exception) {
            debug("Error loading model file: $modelPath - ${e.message}")
            e.printStackTrace()
            throw e
        }
    }

    /**
     * Main detection function that processes an image and returns detected objects
     */
    fun detect(bitmap: Bitmap, confidenceThreshold: Float = CONFIDENCE_THRESHOLD,
               iouThreshold: Float = IOU_THRESHOLD): List<Detection> {
        val startTime = SystemClock.elapsedRealtime()
        debug("Starting detection with conf=$confidenceThreshold, iou=$iouThreshold")

        try {
            // Add debug for input dimensions
            debug("Input image dimensions: ${bitmap.width}x${bitmap.height}")

            // Convert Bitmap to Mat for OpenCV processing
            val inputMat = Mat()
            Utils.bitmapToMat(bitmap, inputMat)
            Imgproc.cvtColor(inputMat, inputMat, Imgproc.COLOR_RGBA2BGR)

            // Prepare input for TFLite
            val originalSize = Size(bitmap.width.toDouble(), bitmap.height.toDouble())
            val resizedImgMat = Mat() // Will hold the resized image

            // Input shape for model
            val modelInputShape = Size(inputWidth.toDouble(), inputHeight.toDouble())
            debug("Model input shape: ${modelInputShape.width.toInt()}x${modelInputShape.height.toInt()}")

            // First preprocess using OpenCV
            val inputTensor = preprocessImageOpenCV(
                inputMat,
                resizedImgMat,
                modelInputShape
            )

            // Run inference
            return try {
                val outputs = runInference(inputTensor)

                // Process outputs to get detections
                val detections = postprocess(
                    outputs,
                    originalSize,
                    Size(inputWidth.toDouble(), inputHeight.toDouble()),
                    confidenceThreshold,
                    iouThreshold
                )

                val inferenceTime = SystemClock.elapsedRealtime() - startTime
                debug("Detection completed in $inferenceTime ms with ${detections.size} objects")

                detections
            } catch (e: Exception) {
                debug("Error during inference: ${e.message}")
                e.printStackTrace()
                emptyList() // Return empty list on error
            } finally {
                // Ensure we clean up resources
                inputMat.release()
                resizedImgMat.release()
            }
        } catch (e: Exception) {
            debug("Error preparing input: ${e.message}")
            e.printStackTrace()
            return emptyList()
        }
    }

    /**
     * Preprocess the input image using OpenCV to match the C++ implementation exactly
     * Optimized for image quality preservation
     */
    private fun preprocessImageOpenCV(image: Mat, outImage: Mat, newShape: Size): ByteBuffer {
        val scopedTimer = ScopedTimer("preprocessing")

        // Track original dimensions before any processing
        debug("Original image dimensions: ${image.width()}x${image.height()}")

        try {
            // Check input image validity
            if (image.empty()) {
                debug("Warning: Input image is empty")
                throw IllegalArgumentException("Input image is empty")
            }
            
            // Use high-quality resize with letterboxing to maintain aspect ratio
            letterBox(image, outImage, newShape, Scalar(114.0, 114.0, 114.0), 
                      interpolation = Imgproc.INTER_LINEAR)

            // Log resized dimensions with letterboxing
            debug("After letterbox: ${outImage.width()}x${outImage.height()}")

            // Convert BGR to RGB (YOLOv11 expects RGB input)
            val rgbMat = Mat()
            Imgproc.cvtColor(outImage, rgbMat, Imgproc.COLOR_BGR2RGB)

            // DEBUG: Output dimensions for verification
            debug("Preprocessed image dimensions: ${rgbMat.width()}x${rgbMat.height()}")

            // Prepare the ByteBuffer to store the model input data
            val bytesPerChannel = if (isQuantized) 1 else 4
            val inputBuffer = ByteBuffer.allocateDirect(1 * inputWidth * inputHeight * 3 * bytesPerChannel)
            inputBuffer.order(ByteOrder.nativeOrder())

            // Convert to proper format for TFLite
            if (isQuantized) {
                // For quantized models: use efficient direct buffer copy
                val pixelBuffer = ByteArray(3)
                for (y in 0 until rgbMat.rows()) {
                    for (x in 0 until rgbMat.cols()) {
                        rgbMat.get(y, x, pixelBuffer)
                        // RGB order (as expected by the model)
                        inputBuffer.put(pixelBuffer[0]) // R
                        inputBuffer.put(pixelBuffer[1]) // G
                        inputBuffer.put(pixelBuffer[2]) // B
                    }
                }
            } else {
                // For float models: use optimized normalization
                val normalizedMat = Mat()
                rgbMat.convertTo(normalizedMat, CvType.CV_32FC3, 1.0/255.0)
                
                // Use more efficient pixel-by-pixel access for better performance
                val floatPixels = FloatArray(3)
                for (y in 0 until normalizedMat.rows()) {
                    for (x in 0 until normalizedMat.cols()) {
                        normalizedMat.get(y, x, floatPixels)
                        // RGB order
                        inputBuffer.putFloat(floatPixels[0]) // R
                        inputBuffer.putFloat(floatPixels[1]) // G
                        inputBuffer.putFloat(floatPixels[2]) // B
                    }
                }
                normalizedMat.release()
            }

            inputBuffer.rewind()
            rgbMat.release()
            
            scopedTimer.stop()
            return inputBuffer
        } catch (e: Exception) {
            debug("Error during preprocessing: ${e.message}")
            e.printStackTrace()
            throw e
        }
    }

    /**
     * Runs inference with TFLite and returns the raw output
     */
    private fun runInference(inputBuffer: ByteBuffer): Map<Int, Any> {
        val scopedTimer = ScopedTimer("inference")

        val outputs: MutableMap<Int, Any> = HashMap()

        try {
            // YOLOv11 with TFLite typically outputs a single tensor
            val outputShape = interpreter.getOutputTensor(0).shape()
            debug("Output tensor shape: ${outputShape.joinToString()}")

            // Correctly allocate output buffer based on the shape
            if (isQuantized) {
                val outputSize = outputShape.reduce { acc, i -> acc * i }
                val outputBuffer = ByteBuffer.allocateDirect(4 * outputSize)
                    .order(ByteOrder.nativeOrder())
                outputs[0] = outputBuffer

                // Run inference with quantized model
                interpreter.run(inputBuffer, outputBuffer)
            } else {
                val outputSize = outputShape.reduce { acc, i -> acc * i }
                val outputBuffer = ByteBuffer.allocateDirect(4 * outputSize)
                    .order(ByteOrder.nativeOrder())
                outputs[0] = outputBuffer

                // Run inference with float model
                interpreter.run(inputBuffer, outputBuffer)

                // Debug: Peek at some values to verify output format
                outputBuffer.rewind()
                val values = FloatArray(min(10, outputSize))
                for (i in values.indices) {
                    values[i] = outputBuffer.float
                }
                debug("First few output values: ${values.joinToString()}")
                outputBuffer.rewind()
            }
        } catch (e: Exception) {
            debug("Error during inference: ${e.message}")
            e.printStackTrace()
        }

        scopedTimer.stop()
        return outputs
    }

    /**
     * Post-processes the model outputs to extract detections
     * Modified to correctly handle normalized coordinates
     */
    private fun postprocess(
        outputMap: Map<Int, Any>,
        originalImageSize: Size,
        resizedImageShape: Size,
        confThreshold: Float,
        iouThreshold: Float
    ): List<Detection> {
        val scopedTimer = ScopedTimer("postprocessing")

        val detections = mutableListOf<Detection>()

        try {
            // Get output buffer
            val outputBuffer = outputMap[0] as ByteBuffer
            outputBuffer.rewind()

            // Get output dimensions
            val outputShapes = interpreter.getOutputTensor(0).shape()
            debug("Output tensor shape: ${outputShapes.joinToString()}")

            // YOLOv11 output tensor shape is [1, 84+4, 8400] = [batch, classes+xywh, predictions]
            // This is in TRANSPOSE format (different from YOLOv8)
            val num_classes = outputShapes[1] - 4  // 84 classes (88 - 4)
            val num_predictions = outputShapes[2]   // 8400 predictions

            debug("Processing output tensor: features=${outputShapes[1]}, predictions=$num_predictions, classes=$num_classes")

            // Extract boxes, confidences, and class ids
            val boxes = mutableListOf<RectF>()
            val confidences = mutableListOf<Float>()
            val classIds = mutableListOf<Int>()
            val nmsBoxes = mutableListOf<RectF>() // For class-separated NMS

            // Create a float array from the buffer for more efficient access
            val outputArray = FloatArray(outputShapes[0] * outputShapes[1] * outputShapes[2])
            outputBuffer.rewind()
            for (i in outputArray.indices) {
                outputArray[i] = outputBuffer.float
            }

            // Process each prediction
            for (i in 0 until num_predictions) {
                // Find class with maximum score and its index
                var maxScore = -Float.MAX_VALUE
                var classId = -1

                // Scan through all classes (start at index 4, after x,y,w,h)
                for (c in 0 until num_classes) {
                    // Class scores are after the 4 box coordinates
                    val score = outputArray[(4 + c) * num_predictions + i]
                    if (score > maxScore) {
                        maxScore = score
                        classId = c
                    }
                }

                // Filter by confidence threshold
                if (maxScore >= confThreshold) {
                    // Extract bounding box coordinates (normalized between 0-1)
                    val x = outputArray[0 * num_predictions + i]  // center_x
                    val y = outputArray[1 * num_predictions + i]  // center_y
                    val w = outputArray[2 * num_predictions + i]  // width
                    val h = outputArray[3 * num_predictions + i]  // height

                    // Convert from center format (xywh) to corner format (xyxy) - all normalized
                    val left = x - w / 2
                    val top = y - h / 2
                    val right = x + w / 2
                    val bottom = y + h / 2

                    debug("Detection found: center=($x,$y), wh=($w,$h), score=$maxScore, class=$classId")
                    debug("            box normalized: ($left,$top,$right,$bottom)")

                    // Scale coordinates to original image size
                    val scaledBox = scaleCoords(
                        resizedImageShape,
                        RectF(left, top, right, bottom),
                        originalImageSize
                    )

                    // Additional debug for scaled box
                    debug("            box in original image: (${scaledBox.left},${scaledBox.top},${scaledBox.right},${scaledBox.bottom})")

                    // Validate dimensions before adding
                    val boxWidth = scaledBox.right - scaledBox.left
                    val boxHeight = scaledBox.bottom - scaledBox.top

                    if (boxWidth > 1 && boxHeight > 1) {  // Ensure reasonable size
                        // Round coordinates to integer precision
                        val roundedBox = RectF(
                            round(scaledBox.left),
                            round(scaledBox.top),
                            round(scaledBox.right),
                            round(scaledBox.bottom)
                        )

                        // Create offset box for NMS with class separation
                        val nmsBox = RectF(
                            roundedBox.left + classId * 7680f,
                            roundedBox.top + classId * 7680f,
                            roundedBox.right + classId * 7680f,
                            roundedBox.bottom + classId * 7680f
                        )

                        nmsBoxes.add(nmsBox)
                        boxes.add(roundedBox)
                        confidences.add(maxScore)
                        classIds.add(classId)
                    } else {
                        debug("Skipped detection with invalid dimensions: ${boxWidth}x${boxHeight}")
                    }
                }
            }

            debug("Found ${boxes.size} raw detections before NMS")

            // Run NMS to eliminate redundant boxes
            val selectedIndices = mutableListOf<Int>()
            nonMaxSuppression(nmsBoxes, confidences, confThreshold, iouThreshold, selectedIndices)

            debug("After NMS: ${selectedIndices.size} detections remaining")

            // Create final detection objects
            for (idx in selectedIndices) {
                val box = boxes[idx]

                // Calculate width and height from corners
                val width = box.right - box.left
                val height = box.bottom - box.top

                // Create detection object with proper dimensions
                val detection = Detection(
                    BoundingBox(
                        box.left.toInt(),
                        box.top.toInt(),
                        width.toInt(),
                        height.toInt()
                    ),
                    confidences[idx],
                    classIds[idx]
                )

                detections.add(detection)
                debug("Added detection: box=${detection.box.x},${detection.box.y},${detection.box.width},${detection.box.height}, " +
                        "conf=${detection.conf}, class=${classIds[idx]}")
            }
        } catch (e: Exception) {
            debug("Error during postprocessing: ${e.message}")
            e.printStackTrace()
        }

        scopedTimer.stop()
        return detections
    }

    /**
     * Draws bounding boxes on the provided bitmap
     */
    fun drawDetections(bitmap: Bitmap, detections: List<Detection>): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val paint = Paint()
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = max(bitmap.width, bitmap.height) * 0.004f

        val textPaint = Paint()
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = max(bitmap.width, bitmap.height) * 0.02f

        // Filter detections to ensure quality results
        val filteredDetections = detections.filter {
            it.conf > CONFIDENCE_THRESHOLD &&
                    it.classId >= 0 &&
                    it.classId < classNames.size
        }

        for (detection in filteredDetections) {
            // Get color for this class
            val color = classColors[detection.classId % classColors.size]
            paint.color = Color.rgb(color[0], color[1], color[2])

            // Draw bounding box
            canvas.drawRect(
                detection.box.x.toFloat(),
                detection.box.y.toFloat(),
                (detection.box.x + detection.box.width).toFloat(),
                (detection.box.y + detection.box.height).toFloat(),
                paint
            )

            // Create label text
            val label = "${classNames[detection.classId]}: ${(detection.conf * 100).toInt()}%"

            // Measure text for background rectangle
            val textWidth = textPaint.measureText(label)
            val textHeight = textPaint.textSize

            // Define label position
            val labelY = max(detection.box.y.toFloat(), textHeight + 5f)

            // Draw background rectangle for text
            val bgPaint = Paint()
            bgPaint.color = Color.rgb(color[0], color[1], color[2])
            bgPaint.style = Paint.Style.FILL

            canvas.drawRect(
                detection.box.x.toFloat(),
                labelY - textHeight - 5f,
                detection.box.x.toFloat() + textWidth + 10f,
                labelY + 5f,
                bgPaint
            )

            // Draw text
            textPaint.color = Color.WHITE
            canvas.drawText(
                label,
                detection.box.x.toFloat() + 5f,
                labelY - 5f,
                textPaint
            )
        }

        return mutableBitmap
    }

    /**
     * Draws bounding boxes and semi-transparent masks on the provided bitmap
     */
    fun drawDetectionsMask(bitmap: Bitmap, detections: List<Detection>, maskAlpha: Float = 0.4f): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val width = bitmap.width
        val height = bitmap.height

        // Create a mask bitmap for overlay
        val maskBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val maskCanvas = Canvas(maskBitmap)

        // Filter detections to ensure quality results
        val filteredDetections = detections.filter {
            it.conf > CONFIDENCE_THRESHOLD &&
                    it.classId >= 0 &&
                    it.classId < classNames.size
        }

        // Draw filled rectangles on mask bitmap
        for (detection in filteredDetections) {
            val color = classColors[detection.classId % classColors.size]
            val paint = Paint()
            paint.color = Color.argb(
                (255 * maskAlpha).toInt(),
                color[0],
                color[1],
                color[2]
            )
            paint.style = Paint.Style.FILL

            maskCanvas.drawRect(
                detection.box.x.toFloat(),
                detection.box.y.toFloat(),
                (detection.box.x + detection.box.width).toFloat(),
                (detection.box.y + detection.box.height).toFloat(),
                paint
            )
        }

        // Overlay mask on original image
        val canvas = Canvas(mutableBitmap)
        val paint = Paint()
        paint.alpha = (255 * maskAlpha).toInt()
        canvas.drawBitmap(maskBitmap, 0f, 0f, paint)

        // Draw bounding boxes and labels (reusing existing method but with full opacity)
        val mainCanvas = Canvas(mutableBitmap)
        val boxPaint = Paint()
        boxPaint.style = Paint.Style.STROKE
        boxPaint.strokeWidth = max(width, height) * 0.004f

        val textPaint = Paint()
        textPaint.textSize = max(width, height) * 0.02f

        for (detection in filteredDetections) {
            val color = classColors[detection.classId % classColors.size]
            boxPaint.color = Color.rgb(color[0], color[1], color[2])

            // Draw bounding box
            mainCanvas.drawRect(
                detection.box.x.toFloat(),
                detection.box.y.toFloat(),
                (detection.box.x + detection.box.width).toFloat(),
                (detection.box.y + detection.box.height).toFloat(),
                boxPaint
            )

            // Create and draw label
            val label = "${classNames[detection.classId]}: ${(detection.conf * 100).toInt()}%"
            val textWidth = textPaint.measureText(label)
            val textHeight = textPaint.textSize

            val labelY = max(detection.box.y.toFloat(), textHeight + 5f)

            val bgPaint = Paint()
            bgPaint.color = Color.rgb(color[0], color[1], color[2])
            bgPaint.style = Paint.Style.FILL

            mainCanvas.drawRect(
                detection.box.x.toFloat(),
                labelY - textHeight - 5f,
                detection.box.x.toFloat() + textWidth + 10f,
                labelY + 5f,
                bgPaint
            )

            textPaint.color = Color.WHITE
            mainCanvas.drawText(
                label,
                detection.box.x.toFloat() + 5f,
                labelY - 5f,
                textPaint
            )
        }

        // Clean up
        maskBitmap.recycle()

        return mutableBitmap
    }

    /**
     * Loads class names from a file
     */
    private fun loadClassNames(labelsPath: String): List<String> {
        return context.assets.open(labelsPath).bufferedReader().useLines {
            it.map { line -> line.trim() }.filter { it.isNotEmpty() }.toList()
        }
    }

    /**
     * Generate colors for visualization
     */
    private fun generateColors(numClasses: Int): List<IntArray> {
        val colors = mutableListOf<IntArray>()
        val random = Random(42) // Fixed seed for reproducibility

        for (i in 0 until numClasses) {
            val color = intArrayOf(
                random.nextInt(256),  // R
                random.nextInt(256),  // G
                random.nextInt(256)   // B
            )
            colors.add(color)
        }

        return colors
    }

    /**
     * Get class name for a given class ID
     * @param classId The class ID to get the name for
     * @return The class name or "Unknown" if the ID is invalid
     */
    fun getClassName(classId: Int): String {
        return if (classId >= 0 && classId < classNames.size) {
            classNames[classId]
        } else {
            "Unknown"
        }
    }

    /**
     * Get details about the model's input requirements
     * @return String containing shape and data type information
     */
    fun getInputDetails(): String {
        val inputTensor = interpreter.getInputTensor(0)
        val shape = inputTensor.shape()
        val type = when(inputTensor.dataType()) {
            org.tensorflow.lite.DataType.FLOAT32 -> "FLOAT32"
            org.tensorflow.lite.DataType.UINT8 -> "UINT8"
            else -> "OTHER"
        }
        return "Shape: ${shape.joinToString()}, Type: $type"
    }

    /**
     * Cleanup resources when no longer needed
     */
    fun close() {
        try {
            interpreter.close()
            debug("TFLite interpreter closed")
        } catch (e: Exception) {
            debug("Error closing interpreter: ${e.message}")
        }

        try {
            gpuDelegate?.close()
            debug("GPU delegate resources released")
        } catch (e: Exception) {
            debug("Error closing GPU delegate: ${e.message}")
        }

        gpuDelegate = null
    }

    /**
     * Data classes for detections and bounding boxes
     */
    data class BoundingBox(val x: Int, val y: Int, val width: Int, val height: Int)

    data class Detection(val box: BoundingBox, val conf: Float, val classId: Int)

    /**
     * Helper functions
     */

    /**
     * Letterbox an image to fit a specific size while maintaining aspect ratio
     * Enhanced for better image quality in real-time processing
     */
    private fun letterBox(
        image: Mat,
        outImage: Mat,
        newShape: Size,
        color: Scalar = Scalar(114.0, 114.0, 114.0),
        auto: Boolean = true,
        scaleFill: Boolean = false,
        scaleUp: Boolean = true,
        stride: Int = 32,
        interpolation: Int = Imgproc.INTER_LINEAR
    ) {
        val originalShape = Size(image.cols().toDouble(), image.rows().toDouble())

        // Calculate ratio to fit the image within new shape
        var ratio = min(
            newShape.height / originalShape.height,
            newShape.width / originalShape.width
        ).toFloat()

        // Prevent scaling up if not allowed
        if (!scaleUp) {
            ratio = min(ratio, 1.0f)
        }

        // Calculate new unpadded dimensions - use ceiling to avoid fractional pixels
        val newUnpadW = kotlin.math.ceil(originalShape.width * ratio).toInt()
        val newUnpadH = kotlin.math.ceil(originalShape.height * ratio).toInt()

        // Calculate padding
        val dw = (newShape.width - newUnpadW).toFloat()
        val dh = (newShape.height - newUnpadH).toFloat()

        // Calculate padding distribution with proper alignment
        val padLeft: Int
        val padRight: Int
        val padTop: Int
        val padBottom: Int

        if (auto) {
            // Improved padding alignment to match model's stride expectations
            val dwHalf = kotlin.math.floor(dw / 2.0).toInt()
            val dhHalf = kotlin.math.floor(dh / 2.0).toInt()

            padLeft = dwHalf
            padRight = (dw - dwHalf).toInt()
            padTop = dhHalf
            padBottom = (dh - dhHalf).toInt()
        } else if (scaleFill) {
            // Scale to fill without maintaining aspect ratio
            padLeft = 0
            padRight = 0
            padTop = 0
            padBottom = 0
            Imgproc.resize(image, outImage, newShape, 0.0, 0.0, interpolation)
            return
        } else {
            // Even padding on all sides
            padLeft = kotlin.math.round(dw / 2).toInt()
            padRight = kotlin.math.round(dw - padLeft).toInt()
            padTop = kotlin.math.round(dh / 2).toInt()
            padBottom = kotlin.math.round(dh - padTop).toInt()
        }

        // Log detailed padding information
        debug("Letterbox: original=${originalShape.width}x${originalShape.height}, " +
                "new=${newUnpadW}x${newUnpadH}, ratio=$ratio")
        debug("Letterbox: padding left=$padLeft, right=$padRight, top=$padTop, bottom=$padBottom")

        // Use higher quality interpolation for resize to preserve details
        Imgproc.resize(
            image,
            outImage,
            Size(newUnpadW.toDouble(), newUnpadH.toDouble()),
            0.0, 0.0,
            interpolation  // Use the specified interpolation method
        )

        // Apply padding to create letterboxed image with border replication for better edge handling
        Core.copyMakeBorder(
            outImage,
            outImage,
            padTop,
            padBottom,
            padLeft,
            padRight,
            Core.BORDER_CONSTANT,  // Use constant color for padding
            color
        )
    }

    /**
     * Scale coordinates from model input size to original image size
     * Fixed vertical positioning issue with letterboxed images
     */
    private fun scaleCoords(
        imageShape: Size,
        coords: RectF,
        imageOriginalShape: Size,
        clip: Boolean = true
    ): RectF {
        // Get dimensions in pixels
        val inputWidth = imageShape.width.toFloat()
        val inputHeight = imageShape.height.toFloat()
        val originalWidth = imageOriginalShape.width.toFloat()
        val originalHeight = imageOriginalShape.height.toFloat()

        // Calculate scaling factor (ratio) between original and input sizes
        val gain = min(inputWidth / originalWidth, inputHeight / originalHeight)

        // Calculate padding needed for letterboxing
        val padX = (inputWidth - originalWidth * gain) / 2.0f
        val padY = (inputHeight - originalHeight * gain) / 2.0f

        // Debug dimensions
        debug("Scale coords: input=${inputWidth}x${inputHeight}, original=${originalWidth}x${originalHeight}")
        debug("Scale coords: gain=$gain, padding=($padX, $padY)")
        debug("Scale coords: input normalized=(${coords.left}, ${coords.top}, ${coords.right}, ${coords.bottom})")

        // Convert normalized coordinates [0-1] to absolute pixel coordinates
        val absLeft = coords.left * inputWidth
        val absTop = coords.top * inputHeight
        val absRight = coords.right * inputWidth
        val absBottom = coords.bottom * inputHeight

        debug("Scale coords: absolute pixels=($absLeft, $absTop, $absRight, $absBottom)")

        // Remove padding and scale back to original image dimensions
        val x1 = (absLeft - padX) / gain
        val y1 = (absTop - padY) / gain
        val x2 = (absRight - padX) / gain
        val y2 = (absBottom - padY) / gain

        debug("Scale coords: output original=($x1, $y1, $x2, $y2)")

        // Create result rectangle
        val result = RectF(x1, y1, x2, y2)

        // Clip to image boundaries if requested
        if (clip) {
            result.left = max(0f, min(result.left, originalWidth))
            result.top = max(0f, min(result.top, originalHeight))
            result.right = max(0f, min(result.right, originalWidth))
            result.bottom = max(0f, min(result.bottom, originalHeight))
        }

        return result
    }

    /**
     * Clamp a value between min and max
     */
    private fun clamp(value: Float, min: Float, max: Float): Float {
        return when {
            value < min -> min
            value > max -> max
            else -> value
        }
    }

    /**
     * Non-Maximum Suppression implementation to filter redundant boxes
     * Updated to exactly match the C++ implementation
     */
    private fun nonMaxSuppression(
        boxes: List<RectF>,
        scores: List<Float>,
        scoreThreshold: Float,
        iouThreshold: Float,
        indices: MutableList<Int>
    ) {
        indices.clear()

        // Early return if no boxes
        if (boxes.isEmpty()) {
            return
        }

        // Create list of indices sorted by score (highest first)
        val sortedIndices = boxes.indices
            .filter { scores[it] >= scoreThreshold }
            .sortedByDescending { scores[it] }

        if (sortedIndices.isEmpty()) {
            return
        }

        // Calculate areas once
        val areas = boxes.map { (it.right - it.left) * (it.bottom - it.top) }

        // Suppression mask
        val suppressed = BooleanArray(boxes.size) { false }

        // Process boxes in order of decreasing score
        for (i in sortedIndices.indices) {
            val currentIdx = sortedIndices[i]

            if (suppressed[currentIdx]) {
                continue
            }

            // Add current box to valid detections
            indices.add(currentIdx)

            // Get current box coordinates
            val currentBox = boxes[currentIdx]
            val x1Max = currentBox.left
            val y1Max = currentBox.top
            val x2Max = currentBox.right
            val y2Max = currentBox.bottom
            val areaCurrent = areas[currentIdx]

            // Compare with remaining boxes
            for (j in i + 1 until sortedIndices.size) {
                val compareIdx = sortedIndices[j]

                if (suppressed[compareIdx]) {
                    continue
                }

                // Calculate intersection
                val compareBox = boxes[compareIdx]
                val x1 = max(x1Max, compareBox.left)
                val y1 = max(y1Max, compareBox.top)
                val x2 = min(x2Max, compareBox.right)
                val y2 = min(y2Max, compareBox.bottom)

                val interWidth = max(0f, x2 - x1)
                val interHeight = max(0f, y2 - y1)

                if (interWidth <= 0 || interHeight <= 0) {
                    continue
                }

                val intersection = interWidth * interHeight
                val unionArea = areaCurrent + areas[compareIdx] - intersection
                val iou = if (unionArea > 0) intersection / unionArea else 0f

                // Suppress if IoU exceeds threshold
                if (iou > iouThreshold) {
                    suppressed[compareIdx] = true
                }
            }
        }
    }

    /**
     * Debug print function with enhanced logging
     */
    private fun debug(message: String) {
        Log.d(TAG, message)
        if (BuildConfig.DEBUG) {
            println("YOLO11Detector: $message")
        }
    }

    // Add ScopedTimer implementation (if missing)
    private class ScopedTimer(private val name: String) {
        private val startTime = SystemClock.elapsedRealtime()

        fun stop() {
            val endTime = SystemClock.elapsedRealtime()
//            debug("$name took ${endTime - startTime} ms")
        }
    }
}
