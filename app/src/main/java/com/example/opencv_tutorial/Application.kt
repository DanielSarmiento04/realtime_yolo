package com.example.opencv_tutorial


import android.app.Application
import android.util.Log
import org.opencv.android.OpenCVLoader


/**
 * Application class that initializes OpenCV at app startup
 */
class YoloApplication : Application() {

    override fun onCreate() {
        super.onCreate()

        // Initialize OpenCV with static initialization
        try {
            if (!OpenCVLoader.initDebug()) {
                Log.e(TAG, "OpenCV initialization failed")
            } else {
                Log.i(TAG, "OpenCV initialization succeeded")
                // Load the native library
                System.loadLibrary("opencv_java4")
                Log.i(TAG, "OpenCV native library loaded")
            }
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Failed to load OpenCV native library", e)
        } catch (e: Exception) {
            Log.e(TAG, "Error during OpenCV initialization", e)
        }
    }

    companion object {
        private const val TAG = "YoloApplication"
    }
}
