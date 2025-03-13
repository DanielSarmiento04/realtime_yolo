package com.example.opencv_tutorial

import android.os.SystemClock
import android.util.Log
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.roundToInt

/**
 * Utility class to monitor and track performance metrics for the application
 */
class PerformanceMonitor(private val tag: String) {

    // Tracks each phase of processing
    private val phaseTimes = mutableMapOf<String, MovingAverage>()
    
    // Count of frames processed
    private val frameCount = AtomicInteger(0)
    
    // Global start time
    private val startTime = SystemClock.elapsedRealtime()
    
    // Current phase tracker
    private var currentPhase: String? = null
    private var currentPhaseStart: Long = 0
    
    // Custom phases for tracking
    companion object {
        const val PHASE_PREPROCESSING = "preprocessing"
        const val PHASE_INFERENCE = "inference"
        const val PHASE_POSTPROCESSING = "postprocessing"
        const val PHASE_RENDERING = "rendering"
        const val PHASE_TOTAL = "total_processing"
        
        private const val DEFAULT_WINDOW_SIZE = 20
    }
    
    /**
     * Start tracking a new phase
     */
    fun startPhase(phase: String) {
        // If a phase is already in progress, end it first
        currentPhase?.let { endPhase() }
        
        currentPhase = phase
        currentPhaseStart = SystemClock.elapsedRealtime()
    }
    
    /**
     * End the current phase and record the time
     */
    fun endPhase(): Long {
        currentPhase?.let { phase ->
            val elapsed = SystemClock.elapsedRealtime() - currentPhaseStart
            
            // Get or create the moving average for this phase
            val average = phaseTimes.getOrPut(phase) {
                MovingAverage(DEFAULT_WINDOW_SIZE)
            }
            
            // Update the average
            average.add(elapsed)
            
            // Reset current phase
            currentPhase = null
            
            return elapsed
        }
        return 0
    }
    
    /**
     * Log a completed frame
     */
    fun frameCompleted() {
        val frames = frameCount.incrementAndGet()
        
        // Log performance metrics every 30 frames
        if (frames % 30 == 0) {
            logPerformanceMetrics()
        }
    }
    
    /**
     * Log current performance metrics
     */
    private fun logPerformanceMetrics() {
        val sb = StringBuilder()
        sb.appendLine("=== Performance Metrics ===")
        
        phaseTimes.forEach { (phase, average) ->
            val avgTime = average.getAverage()
            sb.appendLine("$phase: ${avgTime.roundToInt()} ms")
            
            // Calculate FPS for total processing
            if (phase == PHASE_TOTAL && avgTime > 0) {
                val fps = 1000.0 / avgTime
                sb.appendLine("FPS: ${String.format("%.1f", fps)}")
            }
        }
        
        // Calculate overall stats
        val totalElapsed = SystemClock.elapsedRealtime() - startTime
        val totalFrames = frameCount.get()
        val overallFps = totalFrames * 1000.0 / totalElapsed
        
        sb.appendLine("Total frames: $totalFrames")
        sb.appendLine("Running time: ${totalElapsed / 1000} seconds")
        sb.appendLine("Overall FPS: ${String.format("%.1f", overallFps)}")
        
        // Log the stats
        Log.i(tag, sb.toString())
    }
    
    /**
     * Get average time for a specific phase
     */
    fun getAverageTime(phase: String): Double {
        return phaseTimes[phase]?.getAverage() ?: 0.0
    }
    
    /**
     * Reset all metrics
     */
    fun reset() {
        phaseTimes.clear()
        frameCount.set(0)
    }
    
    /**
     * Helper class for calculating moving averages
     */
    private class MovingAverage(private val windowSize: Int) {
        private val values = ArrayDeque<Long>(windowSize)
        
        fun add(value: Long) {
            if (values.size >= windowSize) {
                values.removeFirst()
            }
            values.addLast(value)
        }
        
        fun getAverage(): Double {
            if (values.isEmpty()) return 0.0
            return values.sum().toDouble() / values.size
        }
    }
    
    /**
     * Utility class for automatic timing of code blocks
     */
    inner class ScopedTimer(private val phase: String) : AutoCloseable {
        init {
            startPhase(phase)
        }
        
        override fun close() {
            endPhase()
        }
    }
}
