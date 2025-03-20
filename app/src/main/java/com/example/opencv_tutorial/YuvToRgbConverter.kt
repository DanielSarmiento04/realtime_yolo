package com.example.opencv_tutorial

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.media.Image
import android.renderscript.*
import android.util.Log
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer

/**
 * Helper class for YUV_420_888 to RGB conversion using RenderScript (for API < 31)
 * or Android's YuvImage for API 31+ where RenderScript is deprecated
 */
class YuvToRgbConverter(context: Context) {
    private var rs: RenderScript? = null
    private var scriptYuvToRgb: ScriptIntrinsicYuvToRGB? = null
    private var yuvBuffer: ByteBuffer? = null
    private var inputAllocation: Allocation? = null
    private var outputAllocation: Allocation? = null
    private var isRenderScriptSupported = false
    private val TAG = "YuvToRgbConverter"
    private val appContext = context.applicationContext

    init {
        // Check if RenderScript is available on this device
        try {
            rs = RenderScript.create(appContext)
            scriptYuvToRgb = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))
            isRenderScriptSupported = true
            Log.d(TAG, "RenderScript initialization successful")
        } catch (e: Exception) {
            Log.w(TAG, "RenderScript initialization failed: ${e.message}")
            isRenderScriptSupported = false
        }
    }

    @Suppress("DEPRECATION")
    fun yuvToRgb(image: ImageProxy, output: Bitmap) {
        // Try YuvImage method first (works on all Android versions)
        try {
            val yuvImage = imageToYuv(image)
            val out = java.io.ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
            val imageBytes = out.toByteArray()
            val converted = android.graphics.BitmapFactory.decodeByteArray(
                imageBytes, 0, imageBytes.size)
            
            // Copy the result to output bitmap
            android.graphics.Canvas(output).drawBitmap(converted, 0f, 0f, null)
            converted.recycle()
            Log.d(TAG, "YuvImage conversion successful")
            return
        } catch (e: Exception) {
            Log.w(TAG, "YuvImage method failed: ${e.message}, trying RenderScript")
        }

        // Fall back to RenderScript if available
        if (!isRenderScriptSupported) {
            Log.e(TAG, "Neither YuvImage nor RenderScript methods are available")
            return
        }
        
        // Use RenderScript for conversion
        val width = image.width
        val height = image.height

        // Reuse the allocated buffers if possible
        val yuvBytes = getYuvBytes(image)
        val yuvSize = yuvBytes.size

        if (yuvBuffer == null) {
            yuvBuffer = ByteBuffer.allocateDirect(yuvSize)
        } else if (yuvBuffer!!.capacity() < yuvSize) {
            // Reallocate if needed
            yuvBuffer = ByteBuffer.allocateDirect(yuvSize)
        }
        yuvBuffer!!.clear()
        yuvBuffer!!.put(yuvBytes)
        yuvBuffer!!.rewind()

        val typeBuilder = Type.Builder(rs, Element.U8(rs)).setX(yuvSize)
        val inputType = typeBuilder.create()
        
        if (inputAllocation == null || inputAllocation!!.type != inputType) {
            inputAllocation?.destroy()
            inputAllocation = Allocation.createTyped(rs, inputType)
        }

        if (outputAllocation == null || outputAllocation!!.type.x != output.width || 
            outputAllocation!!.type.y != output.height) {
            outputAllocation?.destroy()
            outputAllocation = Allocation.createFromBitmap(rs, output)
        }

        inputAllocation?.copyFrom(yuvBuffer!!)
        
        scriptYuvToRgb?.setInput(inputAllocation)
        scriptYuvToRgb?.forEach(outputAllocation)
        
        outputAllocation?.copyTo(output)
    }

    private fun getYuvBytes(image: ImageProxy): ByteArray {
        val planes = image.planes
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]

        // Calculate buffer sizes
        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        // Allocate a buffer big enough for all YUV data
        val nv21 = ByteArray(ySize + uSize + vSize)

        // Copy Y plane as-is
        yBuffer.get(nv21, 0, ySize)
        
        // Copy U and V planes - they need special handling for pixel stride
        var pos = ySize
        for (row in 0 until image.height / 2) {
            for (col in 0 until image.width / 2) {
                val uvIndex = col * uPlane.pixelStride + row * uPlane.rowStride
                // NV21 format requires VU order
                vBuffer.get(uvIndex, nv21, pos++, 1)
                uBuffer.get(uvIndex, nv21, pos++, 1)
            }
        }
        
        return nv21
    }

    private fun imageToYuv(image: ImageProxy): android.graphics.YuvImage {
        val planes = image.planes
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]
        
        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        // Calculate buffer sizes
        val ySize = yBuffer.remaining()
        val totalSize = image.width * image.height * 3 / 2  // Size for YUV420 format

        // Allocate a buffer big enough for NV21 format
        val nv21 = ByteArray(totalSize)
        
        // Copy Y plane data (identical in YUV420 formats)
        var position = 0
        for (row in 0 until image.height) {
            val bufferPos = row * yPlane.rowStride
            val length = if (image.width < yPlane.rowStride) image.width else yPlane.rowStride
            yBuffer.position(bufferPos)
            yBuffer.get(nv21, position, length)
            position += length
        }
        
        // U and V planes need interleaving in the right order for NV21
        // NV21 format has VU order in the interleaved UV plane
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride
        
        for (row in 0 until image.height / 2) {
            for (col in 0 until image.width / 2) {
                val vuPos = ySize + position
                val uvIndex = col * uvPixelStride + row * uvRowStride
                
                // Get V value first for NV21, then U
                nv21[vuPos] = vBuffer.get(uvIndex)
                nv21[vuPos + 1] = uBuffer.get(uvIndex)
                position += 2
            }
        }

        return android.graphics.YuvImage(
            nv21, 
            android.graphics.ImageFormat.NV21, 
            image.width, 
            image.height, 
            null
        )
    }

    fun close() {
        inputAllocation?.destroy()
        outputAllocation?.destroy()
        rs?.destroy()
        
        inputAllocation = null
        outputAllocation = null
        rs = null
        scriptYuvToRgb = null
        yuvBuffer = null
    }
}
