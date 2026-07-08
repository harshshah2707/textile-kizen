import React, { useState, useMemo } from 'react';
import { 
  ResponsiveContainer, 
  AreaChart, 
  Area, 
  Tooltip, 
  XAxis, 
  YAxis, 
  CartesianGrid 
} from 'recharts';
import { BarChart3, TrendingUp, Cpu, Award } from 'lucide-react';

const COLORS = {
  'Broken stitch': '#f43f5e',
  'hole': '#ff5722',
  'horizontal': '#eab308',
  'lines': '#a855f7',
  'Needle mark': '#10b981',
  'Pinched fabric': '#3b82f6',
  'stain': '#ec4899',
  'Vertical': '#06b6d4'
};

const LABELS = {
  'Broken stitch': 'Broken Stitch',
  'hole': 'Hole',
  'horizontal': 'Horizontal',
  'lines': 'Lines',
  'Needle mark': 'Needle Mark',
  'Pinched fabric': 'Pinched Fabric',
  'stain': 'Stain',
  'Vertical': 'Vertical'
};

function Analytics({ defectHistory, batchStats, metrics, settings }) {
  // Format elapsed time
  const formatTime = (secs) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m}h ${s < 10 ? '0' : ''}${s}m`;
  };

  // 1. Memoized calculation for defect distribution
  const distributionData = useMemo(() => {
    const counts = {};
    Object.keys(LABELS).forEach(k => { counts[k] = 0; });
    defectHistory.forEach(d => {
      if (counts[d.type] !== undefined) {
        counts[d.type]++;
      }
    });

    const total = defectHistory.length || 1;
    const data = Object.keys(counts).map(key => ({
      name: LABELS[key],
      value: counts[key],
      percent: Math.round((counts[key] / total) * 100),
      color: COLORS[key] || '#8B92A9',
      rawType: key
    }));

    return data;
  }, [defectHistory]);

  // 2. Memoized calculation for 60-minute trends
  const trendData = useMemo(() => {
    const now = Date.now();
    const buckets = Array.from({ length: 7 }).map((_, idx) => {
      const minutesAgo = (6 - idx) * 10;
      const label = `${minutesAgo}m`;
      const bucket = {
        label,
        timeLimit: now - minutesAgo * 60 * 1000,
        total: 0
      };
      Object.keys(LABELS).forEach(k => { bucket[k] = 0; });
      return bucket;
    });

    defectHistory.forEach(d => {
      const ageMs = now - d.timestamp;
      const ageMin = ageMs / (1000 * 60);

      if (ageMin <= 60) {
        for (let i = 0; i < buckets.length; i++) {
          const bucketMin = (6 - i) * 10;
          if (ageMin <= bucketMin) {
            buckets[i].total++;
            if (buckets[i][d.type] !== undefined) {
              buckets[i][d.type]++;
            }
            break;
          }
        }
      }
    });

    return buckets;
  }, [defectHistory]);

  const CustomLineTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white border border-slate-200 p-3 rounded-lg text-[10px] font-mono font-medium flex flex-col gap-1.5 shadow-xl">
          <p className="text-slate-800 font-bold font-sans">TIME: {label} ago</p>
          <div className="flex justify-between gap-4 border-b border-slate-100 pb-1">
            <span className="text-slate-500 font-bold">TOTAL DEFECTS:</span>
            <span className="text-cyan font-bold">{payload[0].value}</span>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 w-full">
      
      {/* 1. DEFECT DISTRIBUTION PANEL (Col-span 4) */}
      <div className="lg:col-span-4 glass-panel p-5 rounded-2xl border border-white/5 flex flex-col gap-4">
        <h3 className="text-[10px] font-bold text-gray uppercase tracking-widest flex items-center gap-1.5 border-b border-white/5 pb-2">
          <BarChart3 className="w-3.5 h-3.5 text-cyan" />
          <span>Defect Distribution</span>
        </h3>

        <div className="flex flex-col gap-3 overflow-y-auto max-h-[170px] pr-1">
          {defectHistory.length === 0 ? (
            <div className="flex flex-col items-center justify-center text-gray text-xs gap-1 py-12 font-mono">
              <span>NO DATA RECORDED</span>
            </div>
          ) : (
            distributionData.map((item) => (
              <div key={item.name} className="flex flex-col gap-1 text-[10px] font-mono">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-white">{item.name.toUpperCase()}</span>
                  <span className="text-gray">{item.value} ({item.percent}%)</span>
                </div>
                <div className="w-full h-2 bg-navy/60 rounded-full overflow-hidden border border-white/5">
                  <div 
                    className="h-full rounded-full transition-all duration-300"
                    style={{ 
                      backgroundColor: item.color,
                      width: `${item.percent || 0}%`,
                      boxShadow: `0 0 8px ${item.color}80`
                    }}
                  ></div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* 2. REAL-TIME TREND CHART (Col-span 5) */}
      <div className="lg:col-span-5 glass-panel p-5 rounded-2xl border border-white/5 flex flex-col gap-3">
        <h3 className="text-[10px] font-bold text-gray uppercase tracking-widest flex items-center gap-1.5 border-b border-white/5 pb-2">
          <TrendingUp className="w-3.5 h-3.5 text-cyan" />
          <span>Real-time Trend (Last 60 mins)</span>
        </h3>

        <div className="w-full h-[170px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={trendData} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
              <defs>
                <linearGradient id="colorTotal" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#2563EB" stopOpacity={0.25}/>
                  <stop offset="95%" stopColor="#2563EB" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(15,23,42,0.05)" />
              <XAxis 
                dataKey="label" 
                stroke="rgba(15,23,42,0.4)" 
                tick={{ fontSize: 9, fontFamily: 'IBM Plex Mono' }}
              />
              <YAxis 
                allowDecimals={false} 
                stroke="rgba(15,23,42,0.4)" 
                tick={{ fontSize: 9, fontFamily: 'IBM Plex Mono' }}
              />
              <Tooltip content={<CustomLineTooltip />} />
              <Area 
                type="monotone" 
                dataKey="total" 
                stroke="#2563EB" 
                strokeWidth={2} 
                fillOpacity={1} 
                fill="url(#colorTotal)"
                dot={{ r: 2, stroke: '#3B82F6', strokeWidth: 1, fill: '#FFFFFF' }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 3. HOURLY PRODUCTION STATS (Col-span 3) */}
      <div className="lg:col-span-3 glass-panel p-5 rounded-2xl border border-white/5 flex flex-col gap-4">
        <h3 className="text-[10px] font-bold text-gray uppercase tracking-widest flex items-center gap-1.5 border-b border-white/5 pb-2">
          <Award className="w-3.5 h-3.5 text-cyan" />
          <span>Production Run Details</span>
        </h3>

        <div className="flex flex-col gap-3 font-mono text-[10px]">
          
          <div className="flex items-center justify-between bg-navy/40 p-2.5 rounded-xl border border-white/5">
            <span className="text-gray font-sans font-medium uppercase">Fabric Inspected</span>
            <span className="text-white font-bold">{batchStats.processedMeters.toFixed(1)} m</span>
          </div>

          <div className="flex items-center justify-between bg-navy/40 p-2.5 rounded-xl border border-white/5">
            <span className="text-gray font-sans font-medium uppercase">Defects Logged</span>
            <span className="text-white font-bold">{batchStats.totalDefects} items</span>
          </div>

          <div className="flex items-center justify-between bg-navy/40 p-2.5 rounded-xl border border-white/5">
            <span className="text-gray font-sans font-medium uppercase">Inspection Time</span>
            <span className="text-white font-bold">{formatTime(batchStats.timeElapsed)}</span>
          </div>

          <div className="flex items-center justify-between bg-navy/40 p-2.5 rounded-xl border border-white/5">
            <span className="text-gray font-sans font-medium uppercase">Batch Status</span>
            <div className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-green animate-ping"></span>
              <span className="text-green font-bold uppercase">RUNNING</span>
            </div>
          </div>

        </div>
      </div>

    </div>
  );
}

export default Analytics;
