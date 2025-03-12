package com.example.opencv_tutorial

import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.os.BatteryManager
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Utility for benchmarking YOLOv11 model performance across different hardware backends
 */
class PerformanceBenchmark(
    private val context: Context,
    private val modelPath: String,
    private val labelsPath: String
) {
    companion object {
        private const val TAG = "PerformanceBenchmark"
        private const val WARMUP_RUNS = 3
        private const val BENCHMARK_RUNS = 10
    }

    // Test image cache
    private var testBitmap: Bitmap? = null

    // Results storage
    data class BenchmarkResult(
        val delegateType: DelegateType,
        val averageInferenceMs: Float,
        val minInferenceMs: Float,
        val maxInferenceMs: Float,
        val stdDeviation: Float,
        val memoryUsageMB: Float,
        val initializationTimeMs: Long,
        val successRate: Float,
        val timestamp: Long = System.currentTimeMillis()
    )

    /**
     * Run benchmarks across all available hardware backends
     * @return Map of delegate type to benchmark results
     */
    suspend fun runFullBenchmark(testImage: Bitmap): Map<DelegateType, BenchmarkResult> {
        Log.d(TAG, "Starting full benchmark suite")
        testBitmap = testImage

        val results = mutableMapOf<DelegateType, BenchmarkResult>()

        // Test each delegate type
        val delegateTypes = listOf(
            DelegateType.CPU,
            DelegateType.GPU,
            DelegateType.NNAPI
        )

        for (delegateType in delegateTypes) {
            try {
                // Check if this delegate is available
                val delegateManager = TFLiteDelegateManager(context)
                val isAvailable = when (delegateType) {
                    DelegateType.GPU -> delegateManager.isGpuDelegateAvailable()
                    DelegateType.NNAPI -> delegateManager.isNnApiDelegateAvailable()
                    else -> true // CPU is always available
                }

                if (!isAvailable) {
                    Log.d(TAG, "Skipping $delegateType benchmark - not available on this device")
                    continue
                }

                Log.d(TAG, "Running benchmark for $delegateType")
                val result = benchmarkDelegate(delegateType, testImage)
                results[delegateType] = result

                Log.d(TAG, "Results for $delegateType: ${result.averageInferenceMs}ms avg, " +
                        "${result.minInferenceMs}ms min, ${result.maxInferenceMs}ms max")

            } catch (e: Exception) {
                Log.e(TAG, "Error benchmarking $delegateType: ${e.message}")
            }
        }

        return results
    }

    /**
     * Benchmark a specific delegate type
     */
    private suspend fun benchmarkDelegate(
        delegateType: DelegateType,
        testImage: Bitmap
    ): BenchmarkResult = withContext(Dispatchers.Default) {

        Log.d(TAG, "Initializing detector with $delegateType")
        val startInitTime = SystemClock.elapsedRealtime()

        // Create detector with the specified delegate
        val useGPU = delegateType == DelegateType.GPU

        val detector = try {
            YOLO11Detector(
                context = context,
                modelPath = modelPath,
                labelsPath = labelsPath,
                useGPU = useGPU
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create detector with $delegateType: ${e.message}")
            return@withContext createErrorResult(delegateType)
        }

        val initTime = SystemClock.elapsedRealtime() - startInitTime
        Log.d(TAG, "Initialization took $initTime ms")

        // Measure memory usage before inference
        val memoryBefore = getMemoryInfo()

        try {
            // Warmup runs
            Log.d(TAG, "Performing $WARMUP_RUNS warmup runs")
            for (i in 0 until WARMUP_RUNS) {
                detector.detect(testImage)
            }

            // Benchmark runs
            Log.d(TAG, "Starting $BENCHMARK_RUNS benchmark runs")
            val inferenceTimes = mutableListOf<Long>()
            var successCount = 0

            for (i in 0 until BENCHMARK_RUNS) {
                try {
                    val startTime = SystemClock.elapsedRealtime()

                    val detections = detector.detect(
                        testImage,
                        YOLO11Detector.CONFIDENCE_THRESHOLD,
                        YOLO11Detector.IOU_THRESHOLD
                    )

                    val inferenceTime = SystemClock.elapsedRealtime() - startTime
                    inferenceTimes.add(inferenceTime)

                    // Count as success if we get any detections or if inference completes
                    successCount++

                    Log.d(TAG, "Run $i: $inferenceTime ms, ${detections.size} detections")
                } catch (e: Exception) {
                    Log.e(TAG, "Error during benchmark run $i: ${e.message}")
                }
            }

            // Measure memory after inference
            val memoryAfter = getMemoryInfo()
            val memoryUsage = memoryAfter - memoryBefore

            // Calculate statistics
            if (inferenceTimes.isEmpty()) {
                Log.e(TAG, "No successful inferences for $delegateType")
                return@withContext createErrorResult(delegateType)
            }

            val avgInference = inferenceTimes.average().toFloat()
            val minInference = inferenceTimes.minOrNull()?.toFloat() ?: 0f
            val maxInference = inferenceTimes.maxOrNull()?.toFloat() ?: 0f

            // Calculate standard deviation
            val variance = inferenceTimes.map {
                val diff = it.toFloat() - avgInference
                diff * diff
            }.average()
            val stdDeviation = kotlin.math.sqrt(variance).toFloat()

            val successRate = successCount.toFloat() / BENCHMARK_RUNS

            // Clean up
            detector.close()

            BenchmarkResult(
                delegateType = delegateType,
                averageInferenceMs = avgInference,
                minInferenceMs = minInference,
                maxInferenceMs = maxInference,
                stdDeviation = stdDeviation,
                memoryUsageMB = memoryUsage,
                initializationTimeMs = initTime,
                successRate = successRate
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error during benchmark: ${e.message}")
            detector.close()
            createErrorResult(delegateType)
        }
    }

    /**
     * Create a result object for when benchmarking fails
     */
    private fun createErrorResult(delegateType: DelegateType): BenchmarkResult {
        return BenchmarkResult(
            delegateType = delegateType,
            averageInferenceMs = -1f,
            minInferenceMs = -1f,
            maxInferenceMs = -1f,
            stdDeviation = -1f,
            memoryUsageMB = -1f,
            initializationTimeMs = -1,
            successRate = 0f
        )
    }

    /**
     * Get current memory usage in MB
     */
    private fun getMemoryInfo(): Float {
        val memoryInfo = ActivityManager.MemoryInfo()
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        activityManager.getMemoryInfo(memoryInfo)

        // Return used memory in MB
        val totalMemory = memoryInfo.totalMem / (1024f * 1024f)
        val availableMemory = memoryInfo.availMem / (1024f * 1024f)
        return totalMemory - availableMemory
    }

    /**
     * Get battery consumption during a test (requires permission)
     */
    private fun getBatteryInfo(): Float {
        try {
            val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
            return batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW) / 1000f
        } catch (e: Exception) {
            Log.e(TAG, "Error getting battery info: ${e.message}")
            return -1f
        }
    }

    /**
     * Save benchmark results to CSV file
     */
    fun saveResultsToCsv(results: Map<DelegateType, BenchmarkResult>): File? {
        try {
            // Create file in app's files directory
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val file = File(context.getExternalFilesDir(null), "benchmark_$timestamp.csv")

            FileWriter(file).use { writer ->
                // Write header
                writer.append("Delegate,Avg Inference (ms),Min Inference (ms),Max Inference (ms)," +
                        "StdDev (ms),Memory Usage (MB),Init Time (ms),Success Rate\n")

                // Write each result
                results.forEach { (_, result) ->
                    writer.append("${result.delegateType},")
                    writer.append("${result.averageInferenceMs},")
                    writer.append("${result.minInferenceMs},")
                    writer.append("${result.maxInferenceMs},")
                    writer.append("${result.stdDeviation},")
                    writer.append("${result.memoryUsageMB},")
                    writer.append("${result.initializationTimeMs},")
                    writer.append("${result.successRate}\n")
                }
            }

            Log.d(TAG, "Benchmark results saved to ${file.absolutePath}")
            return file

        } catch (e: Exception) {
            Log.e(TAG, "Error saving benchmark results: ${e.message}")
            return null
        }
    }

    /**
     * Format results as a human-readable string
     */
    fun formatResultsAsString(results: Map<DelegateType, BenchmarkResult>): String {
        val sb = StringBuilder()
        sb.appendLine("YOLOv11 Benchmark Results")
        sb.appendLine("========================")
        sb.appendLine()

        results.entries.sortedBy { it.value.averageInferenceMs }.forEach { (_, result) ->
            sb.appendLine("${result.delegateType} Delegate:")
            sb.appendLine("  Avg Inference: ${String.format("%.2f", result.averageInferenceMs)} ms")
            sb.appendLine("  Range: ${String.format("%.2f", result.minInferenceMs)} - ${String.format("%.2f", result.maxInferenceMs)} ms")
            sb.appendLine("  Std Deviation: ${String.format("%.2f", result.stdDeviation)} ms")
            sb.appendLine("  Memory Usage: ${String.format("%.1f", result.memoryUsageMB)} MB")
            sb.appendLine("  Init Time: ${result.initializationTimeMs} ms")
            sb.appendLine("  Success Rate: ${String.format("%.1f", result.successRate * 100)}%")
            sb.appendLine()
        }

        return sb.toString()
    }
}
