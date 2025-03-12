package com.example.opencv_tutorial

import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.opencv.android.OpenCVLoader
import java.io.IOException
import java.util.concurrent.Executors
import android.os.SystemClock
import android.graphics.Matrix
import android.os.Build
import androidx.core.content.ContextCompat
import org.tensorflow.lite.gpu.CompatibilityList
import java.util.Locale

class MainActivity : AppCompatActivity() {

    // Views for UI
    private lateinit var imageView: ImageView
    private lateinit var resultText: TextView

    // YOLOv11 detector instance
    private lateinit var yoloDetector: YOLO11Detector

    // Background thread for async loading
    private val backgroundExecutor = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize UI components
        imageView = findViewById(R.id.imageView)
        resultText = findViewById(R.id.resultText)

        // Initialize OpenCV and proceed with detection in background
        initializeOpenCVAndDetector()
    }

    private fun initializeOpenCVAndDetector() {
        resultText.text = "Initializing OpenCV..."

        backgroundExecutor.execute {
            try {
                // Use static initialization for OpenCV
                if (!OpenCVLoader.initDebug()) {
                    Log.e(TAG, "Unable to load OpenCV")
                    runOnUiThread {
                        resultText.text = "Error: OpenCV initialization failed."
                    }
                    return@execute
                }

                // Load native OpenCV library
                try {
                    System.loadLibrary("opencv_java4")
                    Log.i(TAG, "OpenCV loaded successfully")

                    // Now proceed with detector initialization
                    initializeDetectorAndProcess()
                } catch (e: UnsatisfiedLinkError) {
                    Log.e(TAG, "Unable to load OpenCV native library", e)
                    runOnUiThread {
                        resultText.text = "Error: OpenCV native library failed to load.\nError: ${e.message}"
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error during OpenCV initialization", e)
                    runOnUiThread {
                        resultText.text = "Error: ${e.message}"
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Unexpected error during initialization", e)
                runOnUiThread {
                    resultText.text = "Unexpected error: ${e.message}"
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        // Reinitialize if necessary but avoid duplicate initialization
        if (!::yoloDetector.isInitialized && !backgroundExecutor.isShutdown) {
            initializeOpenCVAndDetector()
        }
    }

    private fun initializeDetectorAndProcess() {
        runOnUiThread {
            resultText.text = "Loading model and preparing detection..."
        }

        try {
            // Initialize the YOLO11 detector with model and labels from assets
            // Try alternative model formats if the default fails
            val modelVariants = listOf(
                "best_float16.tflite",  // Try float16 first (smaller, works on many devices)
                "best_float32.tflite",  // Try float32 as fallback (more compatible but larger)
                "best.tflite"           // Try default naming as last resort
            )

            val labelsPath = "classes.txt"

            // Check device compatibility first with more accurate detection
            val useGPU = checkGpuCompatibility()
            Log.d(TAG, "GPU acceleration decision: $useGPU")

            // Try model variants in sequence until one works
            var lastException: Exception? = null
            var detector: YOLO11Detector? = null

            for (modelFile in modelVariants) {
                try {
                    Log.d(TAG, "Attempting to load model: $modelFile")

                    // Check if file exists in assets
                    try {
                        assets.open(modelFile).close()
                    } catch (e: IOException) {
                        Log.d(TAG, "Model file $modelFile not found in assets, skipping")
                        continue
                    }

                    runOnUiThread {
                        resultText.text = "Loading model: $modelFile..."
                    }

                    // Create detector with current model variant
                    detector = YOLO11Detector(
                        context = this,
                        modelPath = modelFile,
                        labelsPath = labelsPath,
                        useGPU = useGPU
                    )

                    // If we get here, initialization succeeded
                    yoloDetector = detector
                    Log.d(TAG, "Successfully initialized detector with model: $modelFile")
                    break

                } catch (e: Exception) {
                    Log.e(TAG, "Failed to initialize with model $modelFile: ${e.message}")
                    e.printStackTrace()
                    lastException = e

                    // If this is GPU mode and failed, try again with CPU
                    if (useGPU) {
                        try {
                            Log.d(TAG, "Retrying model $modelFile with CPU only")
                            detector = YOLO11Detector(
                                context = this,
                                modelPath = modelFile,
                                labelsPath = labelsPath,
                                useGPU = false
                            )

                            yoloDetector = detector
                            Log.d(TAG, "Successfully initialized detector with CPU and model: $modelFile")
                            break
                        } catch (cpuEx: Exception) {
                            Log.e(TAG, "CPU fallback also failed for $modelFile: ${cpuEx.message}")
                            cpuEx.printStackTrace()
                        }
                    }
                }
            }

            // Check if any model variant worked
            if (detector == null) {
                throw RuntimeException("Failed to initialize detector with any available model", lastException)
            }

            runOnUiThread {
                resultText.text = "Model loaded successfully, preparing image..."
            }

            // Load test image from assets
            val imageBitmap = loadImageFromAssets("image_2.jpg")

            if (imageBitmap != null) {
                Log.d(TAG, "Image loaded with dimensions: ${imageBitmap.width}x${imageBitmap.height}")

                runOnUiThread {
                    resultText.text = "Running detection..."
                }

                try {
                    val startTime = SystemClock.elapsedRealtime()

                    // Use exactly the same thresholds as in C++
                    val confThreshold = 0.25f
                    val iouThreshold = 0.45f

                    Log.d(TAG, "Starting detection with conf=$confThreshold, iou=$iouThreshold")

                    // Run detection
                    val detections = yoloDetector.detect(
                        bitmap = imageBitmap,
                        confidenceThreshold = confThreshold,
                        iouThreshold = iouThreshold
                    )

                    val inferenceTime = SystemClock.elapsedRealtime() - startTime
                    Log.d(TAG, "Detection completed in $inferenceTime ms, found ${detections.size} objects")

                    // More detailed logging for debugging
                    if (detections.isEmpty()) {
                        Log.d(TAG, "WARNING: No detections found! Check confidence threshold.")
                    } else {
                        // Log first few detections in more detail
                        detections.take(5).forEachIndexed { index, detection ->
                            val className = yoloDetector.getClassName(detection.classId)
                            val box = detection.box
                            Log.d(TAG, "Top detection #$index: $className (${detection.conf}), " +
                                    "box=${box.x},${box.y},${box.width},${box.height}, " +
                                    "area=${box.width * box.height}")
                        }
                    }

                    // Filter by confidence for display purposes
                    val displayThreshold = 0.30f  // Higher threshold just for display
                    val qualityDetections = detections.filter { it.conf > displayThreshold }
                    Log.d(TAG, "After filtering with threshold $displayThreshold: ${qualityDetections.size} detections")

                    // Draw detections with mask overlay for better visualization
                    val resultBitmap = yoloDetector.drawDetectionsMask(imageBitmap, qualityDetections)

                    // Show results in UI
                    runOnUiThread {
                        // Display the image with detections
                        imageView.setImageBitmap(resultBitmap)

                        // Format and display detection results
                        val resultInfo = StringBuilder()
                        resultInfo.append("Detection completed in $inferenceTime ms\n")
                        resultInfo.append("Found ${detections.size} objects (${qualityDetections.size} shown)\n\n")

                        // Display top detections with highest confidence
                        qualityDetections.sortedByDescending { it.conf }
                            .take(5)
                            .forEach { detection ->
                                val className = yoloDetector.getClassName(detection.classId)
                                val confidence = (detection.conf * 100).toInt()
                                resultInfo.append("• $className: ${confidence}%\n")
                            }

                        resultText.text = resultInfo.toString()
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error during detection", e)
                    // Show original image at least
                    val finalImageBitmap = imageBitmap
                    runOnUiThread {
                        resultText.text = "Detection error: ${e.message}\n${e.stackTraceToString().take(200)}..."
                        imageView.setImageBitmap(finalImageBitmap)
                    }
                }
            } else {
                runOnUiThread {
                    resultText.text = "Error: Failed to load image from assets. Please check that image_2.jpg exists in the assets folder."
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in detection process", e)
            runOnUiThread {
                resultText.text = "Error: ${e.message}\n${e.stackTraceToString().take(300)}..."
            }
        }
    }

    /**
     * Check if the device is compatible with GPU acceleration with enhanced detection
     */
    private fun checkGpuCompatibility(): Boolean {
        Log.d(TAG, "Checking GPU compatibility...")

        // Check if GPU delegation is supported
        val compatList = CompatibilityList()
        val isGpuSupported = compatList.isDelegateSupportedOnThisDevice
        Log.d(TAG, "GPU supported according to compatibility list: $isGpuSupported")

        // Check if running on emulator
        val isEmulator = Build.FINGERPRINT.contains("generic") ||
                Build.FINGERPRINT.startsWith("unknown") ||
                Build.MODEL.contains("google_sdk") ||
                Build.MODEL.contains("Emulator") ||
                Build.MODEL.contains("Android SDK")
        Log.d(TAG, "Is emulator: $isEmulator")

        // Check known problematic device models and manufacturers
        val deviceModel = Build.MODEL.toLowerCase(Locale.ROOT)
        val manufacturer = Build.MANUFACTURER.toLowerCase(Locale.ROOT)

        // List of known problematic device patterns
        val problematicPatterns = listOf(
            "mali-g57", "mali-g72", "mali-g52", "mali-g76",  // Some Mali GPUs have TFLite issues
            "adreno 6", "adreno 5",                          // Some older Adreno GPUs
            "mediatek", "mt6", "helio"                        // Some MediaTek chips
        )

        val isProblematicDevice = problematicPatterns.any { pattern ->
            deviceModel.contains(pattern) || manufacturer.contains(pattern)
        }

        Log.d(TAG, "Device details: manufacturer=$manufacturer, model=$deviceModel")
        Log.d(TAG, "Is problematic device: $isProblematicDevice")

        // Check Android version - some versions have known TFLite GPU issues
        val androidVersion = Build.VERSION.SDK_INT
        val isProblematicAndroidVersion = androidVersion < Build.VERSION_CODES.P  // Android 9-

        Log.d(TAG, "Android version: $androidVersion, problematic: $isProblematicAndroidVersion")

        // Check available memory - GPU acceleration needs sufficient memory
        val memoryInfo = ActivityManager.MemoryInfo()
        val activityManager = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        activityManager.getMemoryInfo(memoryInfo)

        val availableMem = memoryInfo.availMem / (1024 * 1024)  // Convert to MB
        val lowMemory = availableMem < 200  // Less than 200MB available

        Log.d(TAG, "Available memory: $availableMem MB, low memory: $lowMemory")

        // Final decision based on all factors
        val shouldUseGpu = isGpuSupported &&
                !isEmulator &&
                !isProblematicDevice &&
                !isProblematicAndroidVersion &&
                !lowMemory

        Log.d(TAG, "Final GPU acceleration decision: $shouldUseGpu")

        return shouldUseGpu
    }

    /**
     * Load an image from the assets folder with proper orientation and error handling
     */
    private fun loadImageFromAssets(fileName: String): Bitmap? {
        return try {
            val startTime = SystemClock.elapsedRealtime()

            assets.open(fileName).use { inputStream ->
                // Load image size first to check dimensions
                val options = BitmapFactory.Options().apply {
                    inJustDecodeBounds = true
                }
                BitmapFactory.decodeStream(inputStream, null, options)
                inputStream.reset()

                // If image is very large, scale it down to avoid memory issues
                val maxDimension = 1920 // Reasonable max size for detection
                val sampleSize = calculateSampleSize(options.outWidth, options.outHeight, maxDimension)

                // Decode with appropriate sample size
                val decodeOptions = BitmapFactory.Options().apply {
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                    inScaled = false
                    inSampleSize = sampleSize
                }

                val bitmap = BitmapFactory.decodeStream(inputStream, null, decodeOptions)

                val loadTime = SystemClock.elapsedRealtime() - startTime
                Log.d(TAG, "Image loaded: ${bitmap?.width}x${bitmap?.height} " +
                        "(original: ${options.outWidth}x${options.outHeight}, " +
                        "sample size: $sampleSize), took $loadTime ms")
                bitmap
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load image '$fileName'", e)
            null
        }
    }

    /**
     * Calculate appropriate sample size for large images
     */
    private fun calculateSampleSize(width: Int, height: Int, maxDimension: Int): Int {
        var sampleSize = 1
        while (width / sampleSize > maxDimension || height / sampleSize > maxDimension) {
            sampleSize *= 2
        }
        return sampleSize
    }

    override fun onDestroy() {
        super.onDestroy()
        // Clean up resources
        if (::yoloDetector.isInitialized) {
            yoloDetector.close()
        }
        // Shutdown executor service
        backgroundExecutor.shutdown()
    }

    companion object {
        private const val TAG = "YOLO11MainActivity"
    }
}
