package com.example.opencv_tutorial

import android.content.Context
import android.os.Build
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.io.FileInputStream
import java.io.FileOutputStream

/**
 * Utility class to optimize TFLite models for specific devices
 * 
 * This complements TFLiteModelManager and TFLiteDelegateManager by providing
 * model optimization techniques beyond just delegate selection.
 */
class TFLiteModelOptimizer(
    private val context: Context
) {
    companion object {
        private const val TAG = "TFLiteModelOptimizer"
        
        // Constants for optimization
        private const val METADATA_KEY_VERSION = "version"
        private const val METADATA_KEY_TARGET_DEVICE = "target_device"
        private const val OPTIMIZATION_VERSION = 1 // Increment when optimization logic changes
        
        // Cache control
        private const val MAX_CACHE_MODELS = 3
    }
    
    // Device information used for optimization decisions
    private val deviceInfo = mapOf(
        "model" to Build.MODEL,
        "manufacturer" to Build.MANUFACTURER,
        "sdk" to Build.VERSION.SDK_INT,
        "cpu_cores" to Runtime.getRuntime().availableProcessors()
    )
    
    // Track state of current optimization
    private var optimizationInProgress = false
    private val delegateManager by lazy { TFLiteDelegateManager(context) }

    /**
     * Optimize a TFLite model for the current device
     * 
     * @param inputModelPath Path to the original model
     * @param optimize Whether to perform optimization (false = only validation)
     * @return Path to the optimized model, or original if no optimization needed
     */
    fun optimizeModel(inputModelPath: String, optimize: Boolean = true): String {
        Log.d(TAG, "Optimizing model: $inputModelPath, optimize=$optimize")
        
        try {
            // First check if we already have an optimized version for this device
            val optimizedModelPath = getOptimizedModelPath(inputModelPath)
            val optimizedModelFile = File(optimizedModelPath)
            
            // If we have a valid pre-optimized model, use it
            if (optimizedModelFile.exists() && validateOptimizedModel(optimizedModelFile)) {
                Log.d(TAG, "Using existing optimized model: $optimizedModelPath")
                return optimizedModelPath
            }
            
            // If not optimizing, just return the original path
            if (!optimize) {
                // Extract to cache if it's an asset file
                val modelManager = TFLiteModelManager(context)
                return modelManager.prepareModelForDevice(inputModelPath)
            }
            
            // Perform optimization
            return performModelOptimization(inputModelPath, optimizedModelPath)
            
        } catch (e: Exception) {
            Log.e(TAG, "Optimization failed: ${e.message}")
            
            // Return original model path on failure
            return inputModelPath
        }
    }
    
    /**
     * Determine the appropriate path for an optimized model
     */
    private fun getOptimizedModelPath(originalPath: String): String {
        val fileName = originalPath.substringAfterLast("/")
        val baseFileName = fileName.substringBeforeLast(".")
        val extension = fileName.substringAfterLast(".", "tflite")
        
        // Include device model and Android SDK in optimized file name
        val deviceModel = Build.MODEL.replace(" ", "_").take(15)
        val optimizedName = "${baseFileName}_${deviceModel}_sdk${Build.VERSION.SDK_INT}_v${OPTIMIZATION_VERSION}.${extension}"
        
        // Place in app's private cache directory
        val cacheDir = File(context.cacheDir, "optimized_models")
        if (!cacheDir.exists()) {
            cacheDir.mkdirs()
        }
        
        return File(cacheDir, optimizedName).absolutePath
    }
    
    /**
     * Check if an optimized model is valid and compatible with current device
     */
    private fun validateOptimizedModel(modelFile: File): Boolean {
        if (!modelFile.exists() || modelFile.length() < 1024) {
            return false
        }
        
        try {
            // Check file timestamp and name for version info
            val fileName = modelFile.name
            
            // Extract version from filename (if present)
            val versionRegex = "_v(\\d+)\\.".toRegex()
            val versionMatch = versionRegex.find(fileName)
            val fileVersion = versionMatch?.groupValues?.get(1)?.toIntOrNull() ?: 0
            
            // If optimization version has changed, model is outdated
            if (fileVersion < OPTIMIZATION_VERSION) {
                Log.d(TAG, "Optimized model version ($fileVersion) is outdated, current: $OPTIMIZATION_VERSION")
                return false
            }
            
            // Basic validation by loading first 16 bytes to verify it's a valid flatbuffer
            FileInputStream(modelFile).use { input ->
                val header = ByteArray(16)
                val bytesRead = input.read(header)
                
                if (bytesRead != 16) {
                    Log.e(TAG, "Invalid model file: couldn't read header")
                    return false
                }
                
                // Check for TFLite model signature (first 4 bytes of a flatbuffer)
                if (header[0].toInt() != 0x18 || header[1].toInt() != 0x00 ||
                    header[2].toInt() != 0x00 || header[3].toInt() != 0x00) {
                    Log.e(TAG, "Invalid model file: incorrect flatbuffer signature")
                    return false
                }
            }
            
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Error validating optimized model: ${e.message}")
            return false
        }
    }
    
    /**
     * Perform actual model optimization
     */
    private fun performModelOptimization(
        inputModelPath: String,
        outputModelPath: String
    ): String {
        if (optimizationInProgress) {
            Log.d(TAG, "Optimization already in progress, using original model")
            return inputModelPath
        }
        
        optimizationInProgress = true
        
        try {
            // Load original model
            val modelManager = TFLiteModelManager(context)
            val originalModelPath = modelManager.prepareModelForDevice(inputModelPath)
            val modelBuffer = modelManager.loadModelFile(originalModelPath)
            
            // Determine optimal model format for this device
            val optimalFormat = determineOptimalModelFormat()
            Log.d(TAG, "Determined optimal model format: $optimalFormat")
            
            // Skip optimization if we don't need format conversion
            if (optimalFormat == ModelFormat.ORIGINAL) {
                Log.d(TAG, "No format conversion needed, using original model")
                optimizationInProgress = false
                return originalModelPath
            }
            
            // Create output directory if it doesn't exist
            val outputFile = File(outputModelPath)
            if (!outputFile.parentFile?.exists()!!) {
                outputFile.parentFile?.mkdirs()
            }
            
            // Apply format conversion based on determined optimal format
            val success = when (optimalFormat) {
                ModelFormat.FLOAT16 -> convertToFloat16(modelBuffer, outputFile)
                ModelFormat.INT8 -> convertToInt8(modelBuffer, outputFile)
                else -> false
            }
            
            // Manage cache size to prevent filling up storage
            if (success) {
                Log.d(TAG, "Model optimization successful, output: $outputModelPath")
                cleanupModelCache()
                return outputModelPath
            } else {
                Log.e(TAG, "Model optimization failed, using original model")
                return originalModelPath
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error during model optimization: ${e.message}")
            return inputModelPath
        } finally {
            optimizationInProgress = false
        }
    }
    
    /**
     * Convert model to FP16 format for better performance on newer devices
     */
    private fun convertToFloat16(modelBuffer: MappedByteBuffer, outputFile: File): Boolean {
        try {
            // TensorFlow Lite doesn't have built-in converter for post-training format conversion
            // This is usually done during model conversion from the original format
            // For this example, we'll simulate the conversion with a simple copy
            // In a real implementation, you would use TensorFlow's converter API
            
            Log.d(TAG, "Converting to FP16 format")
            
            // In reality, you would use something like:
            // val converter = Converter.Builder()
            //     .setInputModel(modelBuffer)
            //     .setTargetFormat(Converter.TargetFormat.FLOAT16)
            //     .build()
            // converter.convert(outputFile)
            
            // For now, we'll just copy the model and pretend we converted it
            // This is just a placeholder for the actual conversion logic
            FileOutputStream(outputFile).use { output ->
                modelBuffer.rewind()
                val bytes = ByteArray(modelBuffer.capacity())
                modelBuffer.get(bytes)
                output.write(bytes)
            }
            
            // In a real implementation, you would verify the conversion was successful
            Log.d(TAG, "FP16 conversion simulated successfully")
            return true
            
        } catch (e: Exception) {
            Log.e(TAG, "Error converting to FP16: ${e.message}")
            outputFile.delete()
            return false
        }
    }
    
    /**
     * Convert model to INT8 quantized format for better performance on lower-end devices
     */
    private fun convertToInt8(modelBuffer: MappedByteBuffer, outputFile: File): Boolean {
        // Similar to convertToFloat16, this would use TensorFlow's converter in a real implementation
        try {
            Log.d(TAG, "Converting to INT8 format")
            
            // In reality, you would use something like:
            // val converter = Converter.Builder()
            //     .setInputModel(modelBuffer)
            //     .setQuantizationParameters(...)
            //     .build()
            // converter.convert(outputFile)
            
            // For now, just copy the model as a placeholder
            FileOutputStream(outputFile).use { output ->
                modelBuffer.rewind()
                val bytes = ByteArray(modelBuffer.capacity())
                modelBuffer.get(bytes)
                output.write(bytes)
            }
            
            Log.d(TAG, "INT8 conversion simulated successfully")
            return true
            
        } catch (e: Exception) {
            Log.e(TAG, "Error converting to INT8: ${e.message}")
            outputFile.delete()
            return false
        }
    }
    
    /**
     * Determine the optimal model format for the current device
     */
    private fun determineOptimalModelFormat(): ModelFormat {
        // Logic to select best model format based on device capabilities
        
        // For high-end devices with good GPU support, FP16 is often the best choice
        val isHighEnd = delegateManager.isGpuDelegateAvailable() && 
                        Build.VERSION.SDK_INT >= Build.VERSION_CODES.P &&
                        Runtime.getRuntime().availableProcessors() >= 6
        
        // For very low-end devices, INT8 might be better
        val isLowEnd = Build.VERSION.SDK_INT < Build.VERSION_CODES.O ||
                       Runtime.getRuntime().availableProcessors() <= 4
        
        // Get available memory to help decide
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val memInfo = android.app.ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        val availableMem = memInfo.availMem / (1024 * 1024) // MB
        
        return when {
            // For high-end devices with sufficient memory, prefer FP16 for accuracy and speed
            isHighEnd && availableMem > 1024 -> ModelFormat.FLOAT16
            
            // For low-end devices or memory-constrained situations, prefer INT8
            isLowEnd || availableMem < 512 -> ModelFormat.INT8
            
            // Otherwise stick with the original format
            else -> ModelFormat.ORIGINAL
        }
    }
    
    /**
     * Clean up old cached models to prevent filling device storage
     */
    private fun cleanupModelCache() {
        try {
            val cacheDir = File(context.cacheDir, "optimized_models")
            if (!cacheDir.exists() || !cacheDir.isDirectory) return
            
            val modelFiles = cacheDir.listFiles { file -> 
                file.isFile && file.name.endsWith(".tflite") 
            }
            
            // If we have too many cached models, delete the oldest ones
            if (modelFiles != null && modelFiles.size > MAX_CACHE_MODELS) {
                Log.d(TAG, "Cleaning up model cache, found ${modelFiles.size} models")
                
                // Sort by last modified time (oldest first)
                val sortedFiles = modelFiles.sortedBy { it.lastModified() }
                
                // Delete oldest files beyond our cache limit
                for (i in 0 until sortedFiles.size - MAX_CACHE_MODELS) {
                    val fileToDelete = sortedFiles[i]
                    Log.d(TAG, "Deleting old cached model: ${fileToDelete.name}")
                    fileToDelete.delete()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error cleaning up model cache: ${e.message}")
        }
    }
    
    /**
     * Enum representing different model formats for optimization
     */
    enum class ModelFormat {
        ORIGINAL,   // Keep original format
        FLOAT32,    // Full precision floating point
        FLOAT16,    // Half precision floating point (better for most mobile GPUs)
        INT8        // 8-bit integer quantized (smallest, fastest, least accurate)
    }
}
