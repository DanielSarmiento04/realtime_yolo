package com.example.opencv_tutorial

import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.Image
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Utility class for image processing operations
 */
object ImageUtils {
    private const val TAG = "ImageUtils"
    
    /**
     * Convert Android Camera2 Image to OpenCV Mat with minimal overhead
     */
    fun imageToMat(image: Image): Mat {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        
        val nv21 = ByteArray(ySize + uSize + vSize)
        
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        
        val yuvMat = Mat(image.height + image.height / 2, image.width, CvType.CV_8UC1)
        yuvMat.put(0, 0, nv21)
        
        val rgbMat = Mat()
        Imgproc.cvtColor(yuvMat, rgbMat, Imgproc.COLOR_YUV2BGR_NV21)
        
        yuvMat.release()
        
        return rgbMat
    }
    
    /**
     * Convert OpenCV Mat to Android Bitmap
     */
    fun matToBitmap(mat: Mat): Bitmap {
        val bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bitmap)
        return bitmap
    }
    
    /**
     * Resize image with letterboxing to maintain aspect ratio
     */
    fun letterboxImage(image: Mat, outMat: Mat, targetSize: Size) {
        letterBox(image, outMat, targetSize)
    }
    
    /**
     * Apply letterboxing to maintain aspect ratio when resizing
     */
    private fun letterBox(
        image: Mat,
        outImage: Mat,
        newShape: Size,
        color: Scalar = Scalar(114.0, 114.0, 114.0),
        auto: Boolean = true,
        scaleFill: Boolean = false,
        scaleUp: Boolean = true,
        stride: Int = 32
    ) {
        val originalShape = Size(image.cols().toDouble(), image.rows().toDouble())
        
        // Calculate ratio
        var ratio = min(
            newShape.height / originalShape.height,
            newShape.width / originalShape.width
        ).toFloat()
        
        // Don't scale up by default
        if (!scaleUp) {
            ratio = min(ratio, 1.0f)
        }
        
        // Compute new unpadded dimensions
        val newUnpadW = (originalShape.width * ratio).roundToInt()
        val newUnpadH = (originalShape.height * ratio).roundToInt()
        
        // Calculate padding
        val dw = (newShape.width - newUnpadW).toInt()
        val dh = (newShape.height - newUnpadH).toInt()
        
        val padLeft: Int
        val padRight: Int
        val padTop: Int
        val padBottom: Int
        
        if (auto) {
            // Auto-padding aligned to stride
            val dwHalf = ((dw % stride) / 2).toInt()
            val dhHalf = ((dh % stride) / 2).toInt()
            
            padLeft = (dw / 2 - dwHalf)
            padRight = (dw / 2 + dwHalf)
            padTop = (dh / 2 - dhHalf)
            padBottom = (dh / 2 + dhHalf)
        } else if (scaleFill) {
            // Scale to fill without maintaining aspect ratio
            Imgproc.resize(image, outImage, newShape)
            return
        } else {
            // Even padding
            padLeft = (dw / 2)
            padRight = (dw - padLeft)
            padTop = (dh / 2)
            padBottom = (dh - padTop)
        }
        
        // Resize
        Imgproc.resize(
            image,
            outImage,
            Size(newUnpadW.toDouble(), newUnpadH.toDouble()),
            0.0, 0.0,
            Imgproc.INTER_LINEAR
        )
        
        // Add padding
        Core.copyMakeBorder(
            outImage,
            outImage,
            padTop,
            padBottom,
            padLeft,
            padRight,
            Core.BORDER_CONSTANT,
            color
        )
    }
    
    /**
     * Scale coordinates from model input size to original image size
     */
    fun scaleCoords(
        imageShape: Size,     // Model input size
        coords: org.opencv.core.Rect,
        originalSize: Size    // Original image size
    ): org.opencv.core.Rect {
        // Get dimensions
        val inputWidth = imageShape.width
        val inputHeight = imageShape.height
        val originalWidth = originalSize.width
        val originalHeight = originalSize.height
        
        // Calculate gain (ratio between original and input sizes)
        val gain = min(
            inputWidth / originalWidth,
            inputHeight / originalHeight
        )
        
        // Calculate padding
        val padX = (inputWidth - originalWidth * gain) / 2.0
        val padY = (inputHeight - originalHeight * gain) / 2.0
        
        // Scale coordinates back to original size
        val x1 = (coords.x - padX) / gain
        val y1 = (coords.y - padY) / gain
        val x2 = ((coords.x + coords.width) - padX) / gain
        val y2 = ((coords.y + coords.height) - padY) / gain
        
        return org.opencv.core.Rect(
            max(0, min(x1.toInt(), originalWidth.toInt() - 1)),
            max(0, min(y1.toInt(), originalHeight.toInt() - 1)),
            max(1, min(x2.toInt(), originalWidth.toInt()) - max(0, min(x1.toInt(), originalWidth.toInt() - 1))),
            max(1, min(y2.toInt(), originalHeight.toInt()) - max(0, min(y1.toInt(), originalHeight.toInt() - 1)))
        )
    }
    
    /**
     * Rotate bitmap if needed based on rotation degrees
     */
    fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) return bitmap
        
        val matrix = Matrix()
        matrix.postRotate(rotationDegrees.toFloat())
        
        return Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
        )
    }
    
    /**
     * Extract a bitmap from a direct ByteBuffer containing RGBA data
     */
    fun byteBufferToBitmap(buffer: ByteBuffer, width: Int, height: Int): Bitmap {
        buffer.rewind()
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(buffer)
        return bitmap
    }
    
    /**
     * Convert a bitmap to a ByteBuffer for model input
     * with normalization [0-1] for float models
     */
    fun bitmapToByteBuffer(
        bitmap: Bitmap,
        imgMean: Float = 0f,
        imgStd: Float = 255f,
        isQuantized: Boolean = false
    ): ByteBuffer {
        val width = bitmap.width
        val height = bitmap.height
        
        val bytesPerChannel = if (isQuantized) 1 else 4
        val channels = 3
        
        val buffer = ByteBuffer.allocateDirect(bytesPerChannel * height * width * channels)
        buffer.order(java.nio.ByteOrder.nativeOrder())
        
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        
        var pixel = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixelValue = pixels[pixel++]
                
                if (isQuantized) {
                    buffer.put((pixelValue shr 16 and 0xFF).toByte())
                    buffer.put((pixelValue shr 8 and 0xFF).toByte())
                    buffer.put((pixelValue and 0xFF).toByte())
                } else {
                    buffer.putFloat(((pixelValue shr 16 and 0xFF) - imgMean) / imgStd)
                    buffer.putFloat(((pixelValue shr 8 and 0xFF) - imgMean) / imgStd)
                    buffer.putFloat(((pixelValue and 0xFF) - imgMean) / imgStd)
                }
            }
        }
        
        buffer.rewind()
        return buffer
    }
}
