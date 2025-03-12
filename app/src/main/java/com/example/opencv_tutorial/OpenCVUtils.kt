package com.example.opencv_tutorial

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.media.Image
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer

/**
 * Utility class for OpenCV-related operations
 */
object OpenCVUtils {

    /**
     * Converts an ImageProxy (YUV_420_888) to an OpenCV Mat
     * @param imageProxy The CameraX ImageProxy
     * @return Mat The OpenCV Mat in BGR format
     */
    @OptIn(ExperimentalGetImage::class)
    fun imageProxyToMat(imageProxy: ImageProxy): Mat {
        val image = imageProxy.image ?: throw IllegalArgumentException("Image proxy has no image")

        // Check if format is supported
        if (image.format != ImageFormat.YUV_420_888) {
            throw IllegalArgumentException("Unsupported image format: ${image.format}")
        }

        // Extract YUV planes
        val planes = image.planes
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]

        // Y plane dimensions (full resolution)
        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        val width = image.width
        val height = image.height

        // Allocate a Mat for the NV21 image
        val nv21Mat = Mat(height + height / 2, width, CvType.CV_8UC1)

        // Convert Y, U, V planes to NV21 format
        val nv21Buffer = ByteArray(width * height * 3 / 2)

        // Copy Y plane
        val yBuffer = yPlane.buffer
        val ySize = width * height
        for (row in 0 until height) {
            val yOffset = row * yRowStride
            val dstOffset = row * width

            // Copy one row of Y data
            for (col in 0 until width) {
                nv21Buffer[dstOffset + col] = yBuffer[yOffset + col * yPixelStride]
            }
        }

        // Copy interleaved U and V (NV21 format has them interleaved)
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        val uvPixelStride = if (uPlane.pixelStride == 2 && vPlane.pixelStride == 2) 1 else 0
        val uvRowStride = uPlane.rowStride

        for (row in 0 until height / 2) {
            for (col in 0 until width / 2) {
                val uvOffset = row * uvRowStride + col * uPlane.pixelStride
                nv21Buffer[ySize + (row * width) + col * 2] = vBuffer[uvOffset]
                nv21Buffer[ySize + (row * width) + col * 2 + 1] = uBuffer[uvOffset]
            }
        }

        // Put the NV21 data into the Mat
        nv21Mat.put(0, 0, nv21Buffer)

        // Convert to BGR format
        val bgrMat = Mat()
        Imgproc.cvtColor(nv21Mat, bgrMat, Imgproc.COLOR_YUV2BGR_NV21)

        // Rotate according to the image rotation
        val rotatedMat = rotateMat(bgrMat, imageProxy.imageInfo.rotationDegrees)

        // Clean up
        nv21Mat.release()
        if (bgrMat != rotatedMat) {
            bgrMat.release()
        }

        return rotatedMat
    }

    /**
     * Rotates a Mat according to the specified angle
     * @param src Source Mat
     * @param angle Rotation angle in degrees (should be multiple of 90)
     * @return Rotated Mat
     */
    private fun rotateMat(src: Mat, angle: Int): Mat {
        if (angle == 0 || angle == 360) return src

        val dst = Mat()

        when (angle) {
            90 -> {
                // 90 degrees clockwise rotation = transpose + flip around y-axis
                org.opencv.core.Core.transpose(src, dst)
                org.opencv.core.Core.flip(dst, dst, 1) // 1 means flipping around y-axis
            }
            180 -> {
                // 180 degrees rotation = flip around both axes
                org.opencv.core.Core.flip(src, dst, -1) // -1 means flipping around both axes
            }
            270 -> {
                // 270 degrees clockwise = transpose + flip around x-axis
                org.opencv.core.Core.transpose(src, dst)
                org.opencv.core.Core.flip(dst, dst, 0) // 0 means flipping around x-axis
            }
            else -> return src  // Unsupported angle, return original
        }

        return dst
    }

    /**
     * Converts a Mat to a Bitmap
     * @param mat Source Mat in BGR format
     * @return Bitmap
     */
    fun matToBitmap(mat: Mat): Bitmap {
        val bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bitmap)
        return bitmap
    }
}
