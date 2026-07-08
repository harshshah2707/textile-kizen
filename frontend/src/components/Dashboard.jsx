import React from 'react';
import { 
  ShieldAlert, 
  Cpu, 
  Database, 
  Thermometer, 
  TrendingUp, 
  Award,
  TrendingDown
} from 'lucide-react';

function Dashboard({ batchStats, metrics, systemStats, wsStatus }) {
  // Format elapsed time
  const formatTime = (secs) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m}m ${s < 10 ? '0' : ''}${s}s`;
  };

  const isPass = metrics.grade === 'FIRST QUALITY';
  
  // Quality gauge calculation
  const radius = 35;
  const strokeWidth = 6;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (metrics.qualityPercentage / 100) * circumference;

  // Rate of defect indicator helper
  const defectsPerMeter = batchStats.processedMeters > 0 
    ? (batchStats.totalDefects / batchStats.processedMeters).toFixed(2)
    : '0.00';

  return (
    <div className="flex flex-col gap-4 h-full text-white">
      
      {/* CARD 1: KEY METRICS */}
      <div className="glass-panel p-4 rounded-2xl border border-white/5 flex flex-col gap-4">
        <h3 className="text-[10px] font-bold text-gray uppercase tracking-widest flex items-center gap-1.5 border-b border-white/5 pb-2">
          <Award className="w-3.5 h-3.5 text-cyan" />
          <span>Grading Summary</span>
        </h3>

        {/* Large Quality Score Display */}
        <div className="flex flex-col items-center py-2 bg-navy/40 rounded-xl border border-white/5 relative overflow-hidden">
          <span className="text-[9px] font-bold text-gray uppercase tracking-wider mb-1">ASTM QUALITY GRADE</span>
          <span className={`text-xs font-extrabold px-3 py-1 rounded-full border ${
            isPass 
              ? 'text-green bg-green/10 border-green/20' 
              : 'text-red bg-red/10 border-red/20 animate-pulse'
          }`}>
            {isPass ? 'PASS - FIRST QUALITY' : 'FAIL - SECOND CLASS'}
          </span>
        </div>

        {/* Metric Grid */}
        <div className="flex flex-col gap-3 font-mono text-xs">
          
          {/* Total Defects */}
          <div className="flex justify-between items-center bg-navy/20 p-2 rounded-lg border border-white/5">
            <span className="text-gray text-[10px] font-sans font-medium uppercase">TOTAL DEFECTS:</span>
            <span className="text-lg font-extrabold text-white">
              {batchStats.totalDefects}
            </span>
          </div>

          {/* Points rate */}
          <div className="flex justify-between items-center bg-navy/20 p-2 rounded-lg border border-white/5">
            <span className="text-gray text-[10px] font-sans font-medium uppercase">POINTS RATE:</span>
            <span className="text-sm font-bold text-white">
              {metrics.pointsPer100m2} <span className="text-[9px] text-gray">/100m²</span>
            </span>
          </div>

          {/* Average Density */}
          <div className="flex justify-between items-center bg-navy/20 p-2 rounded-lg border border-white/5">
            <span className="text-gray text-[10px] font-sans font-medium uppercase">RATE PER METRE:</span>
            <span className="text-sm font-bold flex items-center gap-1">
              {defectsPerMeter} 
              {parseFloat(defectsPerMeter) > 0.5 ? (
                <TrendingUp className="w-3 h-3 text-orange" />
              ) : (
                <TrendingDown className="w-3 h-3 text-green" />
              )}
            </span>
          </div>

          {/* Inspected Length */}
          <div className="flex justify-between items-center bg-navy/20 p-2 rounded-lg border border-white/5">
            <span className="text-gray text-[10px] font-sans font-medium uppercase">INSPECTED LENGTH:</span>
            <span className="text-sm font-bold text-cyan">{batchStats.processedMeters} m</span>
          </div>

        </div>
      </div>

      {/* CARD 2: SYSTEM HEALTH */}
      <div className="glass-panel p-4 rounded-2xl border border-white/5 flex flex-col gap-4">
        <h3 className="text-[10px] font-bold text-gray uppercase tracking-widest flex items-center gap-1.5 border-b border-white/5 pb-2">
          <Cpu className="w-3.5 h-3.5 text-cyan" />
          <span>Host PC Resources</span>
        </h3>

        <div className="flex flex-col gap-3 font-mono text-[10px]">
          
          {/* CPU usage progress */}
          <div>
            <div className="flex justify-between text-gray font-sans font-medium uppercase mb-1">
              <span>PC CPU Load</span>
              <span className="text-white font-mono font-bold">{wsStatus === 'disconnected' ? '--' : `${systemStats.cpu}%`}</span>
            </div>
            <div className="w-full h-1.5 bg-navy/60 rounded-full overflow-hidden border border-white/5">
              <div 
                className="h-full bg-cyan transition-all duration-300" 
                style={{ width: wsStatus === 'disconnected' ? '0%' : `${systemStats.cpu}%` }}
              ></div>
            </div>
          </div>

          {/* RAM usage progress */}
          <div>
            <div className="flex justify-between text-gray font-sans font-medium uppercase mb-1">
              <span>PC RAM Usage</span>
              <span className="text-white font-mono font-bold">{wsStatus === 'disconnected' ? '--' : `${systemStats.ram}%`}</span>
            </div>
            <div className="w-full h-1.5 bg-navy/60 rounded-full overflow-hidden border border-white/5">
              <div 
                className="h-full bg-light-cyan transition-all duration-300" 
                style={{ width: wsStatus === 'disconnected' ? '0%' : `${systemStats.ram}%` }}
              ></div>
            </div>
          </div>

          {/* Disk usage progress */}
          <div>
            <div className="flex justify-between text-gray font-sans font-medium uppercase mb-1">
              <span>PC Disk Usage</span>
              <span className="text-white font-mono font-bold">{wsStatus === 'disconnected' ? '--' : `${systemStats.disk}%`}</span>
            </div>
            <div className="w-full h-1.5 bg-navy/60 rounded-full overflow-hidden border border-white/5">
              <div 
                className="h-full bg-green transition-all duration-300" 
                style={{ width: wsStatus === 'disconnected' ? '0%' : `${systemStats.disk}%` }}
              ></div>
            </div>
          </div>

        </div>
      </div>

      {/* CARD 3: QUALITY Score Circular Gauge */}
      <div className="glass-panel p-4 rounded-2xl border border-white/5 flex flex-col items-center justify-center gap-3">
        <h3 className="w-full text-[10px] font-bold text-gray uppercase tracking-widest flex items-center gap-1.5 border-b border-white/5 pb-2">
          <Database className="w-3.5 h-3.5 text-cyan" />
          <span>Quality Index</span>
        </h3>

        <div className="relative flex items-center justify-center py-2">
          
          {/* Circular SVG Progress */}
          <svg className="w-28 h-28 transform -rotate-90">
            {/* Background circle */}
            <circle
              cx="56"
              cy="56"
              r={radius}
              fill="transparent"
              stroke="rgba(255, 255, 255, 0.05)"
              strokeWidth={strokeWidth}
            />
            {/* Foreground circle */}
            <circle
              cx="56"
              cy="56"
              r={radius}
              fill="transparent"
              stroke={isPass ? '#06D6A0' : '#E63946'}
              strokeWidth={strokeWidth}
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              className="transition-all duration-500 ease-out"
              style={{ filter: `drop-shadow(0 0 4px ${isPass ? '#06D6A0' : '#E63946'})` }}
            />
          </svg>
          
          {/* Text in center */}
          <div className="absolute flex flex-col items-center justify-center font-mono">
            <span className="text-xl font-extrabold text-white">{metrics.qualityPercentage}%</span>
            <span className="text-[7.5px] font-sans font-extrabold tracking-wider text-gray uppercase">INDEX</span>
          </div>

        </div>

        <div className="text-[10px] font-sans text-gray text-center font-medium mt-1 uppercase">
          {isPass ? (
            <span className="text-green font-bold">✓ BATCH QUALITY STABLE</span>
          ) : (
            <span className="text-red font-bold animate-pulse">⚠ CRITICAL TOLERANCE LIMIT EXCEEDED</span>
          )}
        </div>
      </div>

    </div>
  );
}

export default Dashboard;
