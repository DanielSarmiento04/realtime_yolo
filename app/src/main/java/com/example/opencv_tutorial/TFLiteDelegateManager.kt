package com.example.opencv_tutorial

import android.content.Context
import android.os.Build
import android.util.Log
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate

/**
 * Backend delegate types supported by the application
 */
enum class DelegateType {
    CPU,      // Regular CPU execution
    GPU,      // GPU delegate
    NNAPI,    // Neural Network API delegate
    AUTO      // Auto-selection based on device capabilities
}

/**
 * Performance stats from inference
 */
data class InferenceStats(
    val inferenceTimeMs: Long,
    val delegateType: DelegateType,
    val success: Boolean,
    val errorMessage: String? = null,
    val memoryUsage: Long = -1
)

/**
 * Manages TensorFlow Lite delegates for optimized model execution on different hardware
 */
class TFLiteDelegateManager(private val context: Context) {
    companion object {
        private const val TAG = "TFLiteDelegateManager"

        // Constants for NNAPI configuration
        private const val NNAPI_ALLOW_FP16 = true
        private const val NNAPI_EXECUTION_PREFERENCE = NnApiDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER
    }

    // Track current active delegate for resource management
    private var currentDelegate: Delegate? = null
    private var currentDelegateType: DelegateType = DelegateType.CPU

    // GPU compatibility checker
    private val gpuCompatibilityList = CompatibilityList()

    /**
     * Configure interpreter options with the appropriate delegate based on type
     * @param options The interpreter options to configure
     * @param delegateType The delegate type to use
     * @return The chosen delegate type (may differ if requested type is unavailable)
     */
    fun configureTFLiteInterpreter(options: Interpreter.Options, delegateType: DelegateType): DelegateType {
        // Close any existing delegate first to prevent memory leaks
        closeDelegate()

        // Choose appropriate delegate or let the system decide
        val actualDelegateType = when (delegateType) {
            DelegateType.AUTO -> selectOptimalDelegate()
            else -> delegateType
        }

        Log.d(TAG, "Configuring interpreter with $actualDelegateType delegate")

        // Apply the selected delegate configuration
        when (actualDelegateType) {
            DelegateType.GPU -> configureGpuDelegate(options)
            DelegateType.NNAPI -> configureNnApiDelegate(options)
            DelegateType.CPU -> configureCpuDelegate(options)
            else -> configureCpuDelegate(options) // Fallback
        }

        currentDelegateType = actualDelegateType
        return actualDelegateType
    }

    /**
     * Select the best delegate based on device capabilities
     */
    fun selectOptimalDelegate(): DelegateType {
        Log.d(TAG, "Selecting optimal delegate based on device capabilities")

        // First check if GPU is supported
        if (isGpuDelegateAvailable()) {
            Log.d(TAG, "GPU delegate selected as optimal")
            return DelegateType.GPU
        }

        // Then check if NNAPI is available and reasonably modern (Android 10+)
        if (isNnApiDelegateAvailable() && Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            Log.d(TAG, "NNAPI delegate selected as optimal")
            return DelegateType.NNAPI
        }

        // Fallback to CPU
        Log.d(TAG, "CPU delegate selected as optimal")
        return DelegateType.CPU
    }

    /**
     * Configure with GPU delegate
     */
    private fun configureGpuDelegate(options: Interpreter.Options) {
        try {
            if (!isGpuDelegateAvailable()) {
                Log.w(TAG, "GPU delegate requested but not available, falling back to CPU")
                configureCpuDelegate(options)
                currentDelegateType = DelegateType.CPU
                return
            }

            val delegateOptions = GpuDelegate.Options().apply {
                setPrecisionLossAllowed(true)  // Allow reduced precision for better performance
                setQuantizedModelsAllowed(true)  // Support quantized models
            }

            val gpuDelegate = GpuDelegate(delegateOptions)
            options.addDelegate(gpuDelegate)
            currentDelegate = gpuDelegate

            // Add CPU fallback options too in case GPU ops are not all supported
            configureCpuOptions(options)

            Log.d(TAG, "GPU delegate configured successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to configure GPU delegate: ${e.message}")
            configureCpuDelegate(options)
            currentDelegateType = DelegateType.CPU
        }
    }

    /**
     * Configure with NNAPI delegate
     */
    private fun configureNnApiDelegate(options: Interpreter.Options) {
        try {
            if (!isNnApiDelegateAvailable()) {
                Log.w(TAG, "NNAPI delegate requested but not available, falling back to CPU")
                configureCpuDelegate(options)
                currentDelegateType = DelegateType.CPU
                return
            }

            val nnApiOptions = NnApiDelegate.Options().apply {
                setAllowFp16(NNAPI_ALLOW_FP16)
                setExecutionPreference(NNAPI_EXECUTION_PREFERENCE)

                // Use accelerators explicitly on Android 10+ (API 29+)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    setAcceleratorName("") // Empty string = use any available accelerator
                    setUseNnapiCpu(true) // Allow NNAPI CPU as fallback
                }
            }

            val nnApiDelegate = NnApiDelegate(nnApiOptions)
            options.addDelegate(nnApiDelegate)
            currentDelegate = nnApiDelegate

            // Also configure CPU options as fallback
            configureCpuOptions(options)

            Log.d(TAG, "NNAPI delegate configured successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to configure NNAPI delegate: ${e.message}")
            configureCpuDelegate(options)
            currentDelegateType = DelegateType.CPU
        }
    }

    /**
     * Configure with CPU options
     */
    private fun configureCpuDelegate(options: Interpreter.Options) {
        configureCpuOptions(options)
        currentDelegateType = DelegateType.CPU
        Log.d(TAG, "CPU execution configured")
    }

    /**
     * Configure optimal CPU options based on device capabilities
     */
    private fun configureCpuOptions(options: Interpreter.Options) {
        try {
            // Determine optimal thread count
            val cpuCores = Runtime.getRuntime().availableProcessors()
            val optimalThreads = when {
                cpuCores <= 2 -> 1
                cpuCores <= 4 -> 2
                else -> cpuCores / 2  // Use half the cores for better thermal management
            }

            options.setNumThreads(optimalThreads)

            // Enable optimizations
            options.setUseXNNPACK(true)  // Use XNNPACK for CPU acceleration
            options.setAllowFp16PrecisionForFp32(true)
            options.setAllowBufferHandleOutput(true)

            Log.d(TAG, "CPU options configured with $optimalThreads threads")
        } catch (e: Exception) {
            Log.e(TAG, "Error configuring CPU options: ${e.message}")
            // Use safe defaults
            options.setNumThreads(1)
        }
    }

    /**
     * Check if GPU delegate is available on this device
     */
    fun isGpuDelegateAvailable(): Boolean {
        return gpuCompatibilityList.isDelegateSupportedOnThisDevice
    }

    /**
     * Check if NNAPI delegate is available on this device
     */
    fun isNnApiDelegateAvailable(): Boolean {
        // NNAPI was introduced in Android 8.1 (API 27)
        return Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1
    }

    /**
     * Get the current delegate type
     */
    fun getCurrentDelegateType(): DelegateType = currentDelegateType

    /**
     * Free resources used by delegates
     */
    fun closeDelegate() {
        currentDelegate?.let {
            try {
                when (it) {
                    is GpuDelegate -> it.close()
                    is NnApiDelegate -> it.close()
                    // Other delegate types may need special handling
                }
                Log.d(TAG, "Closed delegate of type $currentDelegateType")
            } catch (e: Exception) {
                Log.e(TAG, "Error closing delegate: ${e.message}")
            }
            currentDelegate = null
        }
    }
}
