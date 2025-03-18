package com.example.opencv_tutorial

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import android.widget.ImageButton
import android.widget.TextView
import androidx.cardview.widget.CardView
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.tensorflow.lite.support.metadata.MetadataExtractor
import java.io.IOException
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.*

/**
 * Activity for real-time object detection using the camera
 */
class CameraActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "CameraActivity"
        private const val REQUEST_CAMERA_PERMISSION = 10
        private const val MODEL_PATH = "yolov11.tflite"
    }

    // UI components
    private lateinit var viewFinder: PreviewView
    private lateinit var overlayView: DetectionOverlayView
    private lateinit var statsText: TextView
    private lateinit var controlPanel: CardView
    private lateinit var switchCameraButton: ImageButton
    private lateinit var settingsButton: ImageButton

    // Camera manager
    private lateinit var cameraManager: CameraManager

    // Coroutine scope for UI updates
    private val activityScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    // Performance tracking
    private val fpsTracker = MovingAverageCalculator(20)
    private val modelBenchmark = ModelBenchmark(5) // Benchmark every 5 seconds

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)

        // Initialize OpenCV
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "Error initializing OpenCV")
            Toast.makeText(this, "Error initializing OpenCV", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        // Initialize UI components
        viewFinder = findViewById(R.id.viewFinder)
        overlayView = findViewById(R.id.detection_overlay)
        statsText = findViewById(R.id.statsText)
        controlPanel = findViewById(R.id.control_panel)
        switchCameraButton = findViewById(R.id.switch_camera)
        settingsButton = findViewById(R.id.settings)

        // Set up UI event listeners
        switchCameraButton.setOnClickListener {
            cameraManager.switchCamera()
        }

        settingsButton.setOnClickListener {
            // Show settings dialog
            DetectionSettingsDialog().show(supportFragmentManager, "settings")
        }

        // Initialize UI elements
        updateStatsDisplay(0.0, 0)

        // Load model metadata for display
        activityScope.launch {
            displayModelInfo()
        }

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA_PERMISSION
            )
        }

        // Force screen to stay in portrait mode for consistent detection
        requestedOrientation = android.content.pm.ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
    }

    private fun startCamera() {
        // Initialize camera manager
        cameraManager = CameraManager(this, this, viewFinder)

        // Set up frame processing callback
        cameraManager.onFrameProcessed = { bitmap, detections, processingTime ->
            // Update overlay view with detections
            overlayView.updateOverlay(bitmap, detections)

            // Calculate and update FPS
            fpsTracker.add(processingTime.toDouble())
            val avgFps = 1000.0 / fpsTracker.getAverage()

            // Update UI with stats
            updateStatsDisplay(avgFps, detections.size)

            // Run model benchmark periodically
            modelBenchmark.maybeBenchmark(
                bitmap,
                onComplete = { benchmarkResults ->
                    Log.d(TAG, "Benchmark results: $benchmarkResults")
                    // Could display benchmark results in UI
                }
            )
        }

        // Start camera
        cameraManager.startCamera()
    }

    private fun updateStatsDisplay(fps: Double, detectionCount: Int) {
        val fpsText = String.format("%.1f FPS", fps)
        statsText.text = "$fpsText | $detectionCount objects"
    }

    private fun displayModelInfo() {
        try {
            // Extract model metadata
            val modelMetadata = extractModelMetadata(MODEL_PATH)
            Log.d(TAG, "Model info: $modelMetadata")

            // Update UI with model info (optional)
            // statsText.text = "${statsText.text}\n$modelMetadata"
        } catch (e: Exception) {
            Log.e(TAG, "Error extracting model metadata: ${e.message}")
        }
    }

    private fun extractModelMetadata(modelPath: String): String {
        try {
            // Convert ByteArray to ByteBuffer for the MetadataExtractor
            val byteArray = assets.open(modelPath).readBytes()
            val modelBuffer = ByteBuffer.wrap(byteArray)
            val metadataExtractor = MetadataExtractor(modelBuffer)

            if (!metadataExtractor.hasMetadata()) {
                return "No metadata found"
            }

            val modelMetadata = metadataExtractor.modelMetadata
            val name = modelMetadata?.name() ?: "Unknown"
            val description = modelMetadata?.description() ?: "No description"
            val version = modelMetadata?.version() ?: "Unknown version"

            val builder = StringBuilder()
            builder.append("Model: $name (v$version)\n")
            builder.append("Description: $description\n")

            // Get input/output info
            val inputTensorCount = metadataExtractor.inputTensorCount
            val outputTensorCount = metadataExtractor.outputTensorCount
            builder.append("Inputs: $inputTensorCount, Outputs: $outputTensorCount\n")

            return builder.toString()
        } catch (e: IOException) {
            Log.e(TAG, "Error extracting metadata: ${e.message}")
            return "Error: ${e.message}"
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(
                    this,
                    "Camera permission is required for detection",
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    override fun onDestroy() {
        super.onDestroy()
        if (::cameraManager.isInitialized) {
            cameraManager.shutdown()
        }
        activityScope.cancel()
    }

    // Expose camera manager for settings dialog
    fun getCameraManager(): CameraManager? {
        return if (::cameraManager.isInitialized) cameraManager else null
    }

    override fun onConfigurationChanged(newConfig: android.content.res.Configuration) {
        super.onConfigurationChanged(newConfig)
        // Notify camera manager of configuration changes
        if (::cameraManager.isInitialized) {
            // Allow the camera manager to adjust to orientation changes
            cameraManager.handleOrientationChange(newConfig.orientation)
        }
    }

    /**
     * Helper class for moving average calculations
     */
    private class MovingAverageCalculator(private val windowSize: Int) {
        private val values = ArrayDeque<Double>(windowSize)

        fun add(value: Double) {
            if (values.size >= windowSize) {
                values.removeFirst()
            }
            values.addLast(value)
        }

        fun getAverage(): Double {
            if (values.isEmpty()) return 0.0
            return values.sum() / values.size
        }
    }

    /**
     * Helper class for periodic model benchmarking
     */
    private class ModelBenchmark(private val intervalSeconds: Int) {
        private var lastBenchmarkTime = 0L
        private val benchmarkScope = CoroutineScope(Dispatchers.Default)

        fun maybeBenchmark(bitmap: Bitmap, onComplete: (Map<String, Any>) -> Unit) {
            val currentTime = System.currentTimeMillis()
            if (currentTime - lastBenchmarkTime > intervalSeconds * 1000) {
                lastBenchmarkTime = currentTime
                
                // Run benchmark in background
                benchmarkScope.launch {
                    try {
                        // Perform benchmark operations
                        val results = runBenchmark(bitmap)
                        withContext(Dispatchers.Main) {
                            onComplete(results)
                        }
                    } catch (e: Exception) {
                        Log.e("ModelBenchmark", "Benchmark failed: ${e.message}")
                    }
                }
            }
        }

        private fun runBenchmark(bitmap: Bitmap): Map<String, Any> {
            // Simplified benchmark - just measure inference time
            val results = mutableMapOf<String, Any>()
            
            val startTime = System.currentTimeMillis()
            
            // Run 5 inference iterations
            repeat(5) {
                // This would normally call the detector directly but is a placeholder
                Thread.sleep(50) // Simulate inference time
            }
            
            val totalTime = System.currentTimeMillis() - startTime
            val avgTime = totalTime / 5.0
            
            results["avg_inference_ms"] = avgTime
            results["timestamp"] = SimpleDateFormat("HH:mm:ss", Locale.US).format(Date())
            
            return results
        }
    }
}
