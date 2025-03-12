package com.example.opencv_tutorial

import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Handles robust model loading with fallback strategies
 */
class ModelLoader(private val context: Context) {
    private val tag = "ModelLoader"

    /**
     * Load model with built-in fallback mechanisms
     *
     * @param modelPath Path to the model in assets
     * @return MappedByteBuffer containing the model data
     */
    fun loadModel(modelPath: String): MappedByteBuffer {
        Log.d(tag, "Loading model: $modelPath")

        // Strategy 1: Try direct asset loading (most efficient)
        try {
            Log.d(tag, "Trying direct asset loading")
            val assetFd = context.assets.openFd(modelPath)
            val fileInputStream = assetFd.createInputStream()
            val fileChannel = fileInputStream.channel
            val startOffset = assetFd.startOffset
            val declaredLength = assetFd.declaredLength

            val mappedBuffer = fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                startOffset,
                declaredLength
            )

            Log.d(tag, "Direct asset loading succeeded")
            return mappedBuffer

        } catch (e: Exception) {
            Log.w(tag, "Direct asset loading failed: ${e.message}")
        }

        // Strategy 2: Extract to file then load (more compatible with some devices)
        try {
            Log.d(tag, "Trying file extraction method")
            val modelFile = extractAssetToFile(modelPath)

            val fileChannel = modelFile.inputStream().channel
            val mappedBuffer = fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                0,
                modelFile.length()
            )

            Log.d(tag, "File extraction method succeeded")
            return mappedBuffer

        } catch (e: Exception) {
            Log.e(tag, "File extraction method failed: ${e.message}")
            throw RuntimeException("Failed to load model: ${e.message}", e)
        }
    }

    /**
     * Extract an asset to the app's file storage for more reliable access
     */
    private fun extractAssetToFile(assetPath: String): File {
        val fileName = assetPath.substringAfterLast("/")
        val outputFile = File(context.filesDir, fileName)

        if (outputFile.exists()) {
            outputFile.delete()
        }

        context.assets.open(assetPath).use { input ->
            FileOutputStream(outputFile).use { output ->
                input.copyTo(output)
            }
        }

        Log.d(tag, "Extracted asset to: ${outputFile.absolutePath}, size: ${outputFile.length()} bytes")
        return outputFile
    }

    /**
     * Choose the most appropriate model format based on device capabilities
     *
     * @return Path to the optimal model file for this device
     */
    fun chooseOptimalModelFormat(): String {
        // Check for GPU support
        val compatList = org.tensorflow.lite.gpu.CompatibilityList()
        val gpuSupported = compatList.isDelegateSupportedOnThisDevice

        // Check Android version - newer versions have better FP16 support
        val isNewerAndroid = Build.VERSION.SDK_INT >= 28

        // Check if device is emulator
        val isEmulator = Build.FINGERPRINT.contains("generic") ||
                Build.MODEL.contains("google_sdk") ||
                Build.MODEL.contains("Emulator")

        return when {
            // For emulators, FP32 is often more reliable
            isEmulator -> "best_float32.tflite"

            // For newer devices with GPU support, try FP16
            gpuSupported && isNewerAndroid -> "best_float16.tflite"

            // For newer devices without GPU support, try FP32
            isNewerAndroid -> "best_float32.tflite"

            // For older devices, use INT8 quantized model
            else -> "best_int8.tflite"
        }
    }
}
