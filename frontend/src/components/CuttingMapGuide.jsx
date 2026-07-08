import React, { useState } from 'react';
import { Scissors, Download, FileText, CheckCircle, AlertTriangle, Info } from 'lucide-react';

function CuttingMapGuide({ selectedRollId, historicalRolls }) {
  const [rollDetails, setRollDetails] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeRollId, setActiveRollId] = useState(null);

  // Load roll details (including defects list)
  const loadRollDetails = (rollId) => {
    setLoading(true);
    setActiveRollId(rollId);
    fetch(`/api/rolls/report/${rollId}`)
      .then(res => res.json())
      .then(data => {
        setRollDetails(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to load roll details", err);
        setLoading(false);
      });
  };

  // Helper to generate cutting segment suggestions based on defect locations
  const generateCuttingSegments = (roll, defects) => {
    const length = roll.length_meters || 0;
    if (!defects || defects.length === 0) {
      return [{ start: 0, end: length, status: 'GOOD', reason: 'No defects detected' }];
    }

    // Sort defects by scanned distance (Y coordinate)
    const sortedDefects = [...defects].sort((a, b) => a.distance_meters - b.distance_meters);
    
    const segments = [];
    let currentPos = 0;
    const buffer = 0.3; // 30cm buffer on either side of a defect segment to cut out

    sortedDefects.forEach((d) => {
      const defectPos = d.distance_meters;
      const defectStart = Math.max(0, defectPos - buffer);
      const defectEnd = Math.min(length, defectPos + buffer);
      const defClass = d.defect_type || d.defect_class || 'defect';

      if (defectStart > currentPos) {
        // Clear fabric segment
        segments.push({
          start: currentPos,
          end: defectStart,
          status: 'GOOD',
          reason: 'Clear Fabric Segment'
        });
      }

      // Add defect / cut zone segment
      const lastSeg = segments[segments.length - 1];
      if (lastSeg && lastSeg.status === 'CUT' && defectStart <= lastSeg.end) {
        // Merge adjacent or overlapping cut zones
        lastSeg.end = Math.max(lastSeg.end, defectEnd);
        lastSeg.reason += `, ${defClass}`;
      } else {
        segments.push({
          start: defectStart,
          end: defectEnd,
          status: 'CUT',
          reason: `Reject zone: ${defClass.toUpperCase()} (${d.size_mm ? d.size_mm.toFixed(1) : 10}mm)`
        });
      }
      currentPos = Math.max(currentPos, defectEnd);
    });

    if (currentPos < length) {
      segments.push({
        start: currentPos,
        end: length,
        status: 'GOOD',
        reason: 'Clear Fabric Segment'
      });
    }

    return segments;
  };

  // Render Visual Fabric ribbon map
  const renderVisualMap = (roll, defects, segments) => {
    const widthMm = roll.fabric_width || 1800;
    const lengthM = roll.length_meters || 1.0;
    
    return (
      <div className="relative w-full bg-slate-950/40 rounded-2xl border border-white/5 p-6 flex flex-col gap-4">
        <h4 className="text-[10px] font-bold text-gray uppercase tracking-widest flex items-center gap-1.5 border-b border-white/5 pb-2">
          <Scissors className="w-3.5 h-3.5 text-cyan" />
          <span>Interactive 2D Cutting Layout Map</span>
        </h4>
        
        {/* Visual Map Layout */}
        <div className="relative flex border border-white/10 bg-slate-900/50 rounded-xl overflow-hidden min-h-[420px] max-h-[550px] overflow-y-auto p-4">
          
          {/* Ruler */}
          <div className="w-16 h-full flex flex-col justify-between text-[9px] font-mono text-slate-400 shrink-0 border-r border-white/5 pr-3 select-none">
            <div className="text-left">0.0 m (Start)</div>
            <div className="text-left my-20">{(lengthM / 2).toFixed(1)} m</div>
            <div className="text-left">{lengthM.toFixed(1)} m (End)</div>
          </div>
          
          {/* Fabric Area */}
          <div className="relative flex-1 bg-slate-950/80 border border-white/10 rounded-lg overflow-hidden ml-3" style={{ height: '500px' }}>
            <div className="absolute inset-0 opacity-[0.03] bg-[radial-gradient(#ffffff_1px,transparent_1px)] [background-size:12px_12px] pointer-events-none"></div>
            
            {/* Cut Segments Overlays */}
            {segments.map((seg, idx) => {
              const top_pct = (seg.start / lengthM) * 100;
              const height_pct = ((seg.end - seg.start) / lengthM) * 100;
              
              if (seg.status === 'CUT') {
                return (
                  <div
                    key={`seg-${idx}`}
                    className="absolute left-0 right-0 bg-red/10 border-y border-dashed border-red/40 flex items-center justify-center pointer-events-none"
                    style={{ top: `${top_pct}%`, height: `${height_pct}%` }}
                  >
                    <div className="bg-red/20 text-red px-2 py-0.5 rounded text-[8px] font-bold uppercase tracking-wider border border-red/35 flex items-center gap-1">
                      <Scissors className="w-2.5 h-2.5" /> Cut out
                    </div>
                  </div>
                );
              }
              return null;
            })}

            {/* Defect Bounding Boxes plotted on flat canvas */}
            {defects.map((d) => {
              const defClass = d.defect_type || d.defect_class || 'defect';
              const xMin = d.bbox_x !== undefined ? d.bbox_x : (d.x_min !== undefined ? d.x_min : 2048);
              const bboxW = d.bbox_w !== undefined ? d.bbox_w : (d.x_max !== undefined && d.x_min !== undefined ? (d.x_max - d.x_min) : 300);
              
              // Map camera coordinates (typically X is 0 to 4096 px) to left percentages
              const x_pct = (xMin / 4096) * 100;
              const w_pct = (bboxW / 4096) * 100;
              
              // Map Y (distance_meters) to top percentage
              const y_pct = (d.distance_meters / lengthM) * 100;
              
              let colorClass = 'bg-red border-red shadow-sm';
              if (defClass === 'stain' || defClass === 'Needle mark') {
                colorClass = 'bg-orange border-orange shadow-sm';
              } else if (defClass === 'horizontal' || defClass === 'lines') {
                colorClass = 'bg-yellow-500 border-yellow-500 shadow-sm';
              }

              return (
                <div
                  key={d.id}
                  title={`${defClass.toUpperCase()} at Y: ${d.distance_meters.toFixed(2)}m`}
                  className={`absolute rounded border-2 cursor-pointer transition-transform hover:scale-130 flex items-center justify-center text-[7px] text-white font-extrabold z-10 ${colorClass}`}
                  style={{
                    left: `${Math.max(1, Math.min(90, x_pct))}%`,
                    top: `${Math.max(1, Math.min(98, y_pct))}%`,
                    width: `${Math.max(12, w_pct)}px`,
                    height: '12px'
                  }}
                >
                  {d.id}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start text-white max-w-7xl mx-auto">
      
      {/* LEFT COLUMN: Roll Picker List (Col-span 4) */}
      <div className="xl:col-span-4 glass-panel p-5 rounded-2xl border border-white/5 flex flex-col gap-4 shrink-0">
        <h3 className="text-xs font-bold text-gray uppercase tracking-widest flex items-center gap-1.5 border-b border-white/5 pb-2">
          <FileText className="w-4 h-4 text-cyan" />
          <span>Select Inspection Run</span>
        </h3>
        
        {historicalRolls.length === 0 ? (
          <div className="text-[10px] text-gray uppercase tracking-wider font-mono text-center py-10">
            No logged inspection runs.
          </div>
        ) : (
          <div className="flex flex-col gap-2 max-h-[500px] overflow-y-auto pr-1">
            {historicalRolls.map(r => (
              <button
                key={r.id}
                onClick={() => loadRollDetails(r.id)}
                className={`w-full p-3 rounded-xl border text-left transition-all flex flex-col gap-1.5 cursor-pointer ${
                  activeRollId === r.id
                    ? 'bg-blue-600/10 border-blue-500/50 text-white'
                    : 'bg-navy/30 border-white/5 hover:border-white/20 text-slate-300'
                }`}
              >
                <div className="flex justify-between items-center w-full">
                  <span className="text-xs font-bold font-mono">{r.roll_number}</span>
                  <span className={`px-2 py-0.5 rounded text-[8px] font-extrabold uppercase tracking-widest ${
                    r.grade === 'FIRST QUALITY' ? 'bg-green/15 text-green border border-green/35' : 'bg-red/15 text-red border border-red/35'
                  }`}>
                    {r.grade === 'FIRST QUALITY' ? 'Pass' : 'Fail'}
                  </span>
                </div>
                
                <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-[10px] font-mono text-slate-400">
                  <div>Mat: <span className="text-slate-200">{r.material_name}</span></div>
                  <div>Len: <span className="text-slate-200">{r.length_meters.toFixed(1)}m</span></div>
                  <div>Points: <span className="text-orange font-bold">{r.total_points} pts</span></div>
                  <div>Date: <span className="text-slate-300 text-[9px]">{(r.started_at || r.start_time || '').split(' ')[0]}</span></div>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* RIGHT COLUMN: Interactive Cutting Map Guide (Col-span 8) */}
      <div className="xl:col-span-8 flex flex-col gap-6">
        {loading ? (
          <div className="glass-panel p-20 rounded-2xl border border-white/5 flex items-center justify-center text-xs uppercase tracking-widest text-cyan font-mono">
            Loading cutting layout details...
          </div>
        ) : rollDetails ? (
          <div className="flex flex-col gap-6">
            
            {/* Header statistics summary */}
            <div className="glass-panel p-5 rounded-2xl border border-white/5 flex flex-col md:flex-row justify-between gap-4">
              <div className="flex flex-col gap-1.5">
                <div className="text-[10px] text-gray uppercase tracking-wider font-bold">Active Cutting Pattern</div>
                <div className="text-sm font-extrabold text-white font-mono flex items-center gap-2">
                  <span>{rollDetails.roll.roll_number}</span>
                  <span className="text-xs bg-white/5 px-2 py-0.5 rounded text-gray font-normal">
                    {rollDetails.roll.material_name}
                  </span>
                </div>
                <div className="text-[10px] text-slate-400 font-mono">
                  Scanned on {rollDetails.roll.started_at || rollDetails.roll.start_time} by {rollDetails.roll.operator_name}
                </div>
              </div>

              <div className="flex flex-wrap gap-4 items-center">
                <div className="flex flex-col font-mono">
                  <span className="text-[8px] text-gray uppercase font-bold">Scanned Length</span>
                  <span className="text-xs text-white font-extrabold">{rollDetails.roll.length_meters.toFixed(2)} m</span>
                </div>

                <div className="flex flex-col font-mono">
                  <span className="text-[8px] text-gray uppercase font-bold">Penalty Points</span>
                  <span className="text-xs text-orange font-extrabold">{rollDetails.roll.total_points} pts</span>
                </div>

                <div className="flex flex-col font-mono">
                  <span className="text-[8px] text-gray uppercase font-bold">Verdict Grade</span>
                  <span className={`text-xs font-black ${
                    rollDetails.roll.grade === 'FIRST QUALITY' ? 'text-green' : 'text-red'
                  }`}>
                    {rollDetails.roll.grade}
                  </span>
                </div>

                <a
                  href={`/api/rolls/report/${rollDetails.roll.id}/pdf`}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-[10px] font-bold uppercase tracking-wider flex items-center gap-1.5 transition-all shadow-sm cursor-pointer"
                >
                  <Download className="w-3.5 h-3.5" /> PDF
                </a>
              </div>
            </div>

            {/* Visual flat fabric ribbon rendering */}
            {renderVisualMap(
              rollDetails.roll,
              rollDetails.defects,
              generateCuttingSegments(rollDetails.roll, rollDetails.defects)
            )}

            {/* Detailed cutting guidelines */}
            <div className="glass-panel p-5 rounded-2xl border border-white/5 flex flex-col gap-3">
              <h4 className="text-[10px] font-bold text-gray uppercase tracking-widest flex items-center gap-1.5 border-b border-white/5 pb-2">
                <Info className="w-3.5 h-3.5 text-cyan" />
                <span>Optimal Cutting Instructions Guide</span>
              </h4>
              
              <div className="flex flex-col gap-2 font-mono text-[11px]">
                {generateCuttingSegments(rollDetails.roll, rollDetails.defects).map((seg, idx) => (
                  <div
                    key={`guide-${idx}`}
                    className={`p-3 rounded-xl border flex justify-between items-center ${
                      seg.status === 'GOOD'
                        ? 'bg-green/5 border-green/20 text-slate-200'
                        : 'bg-red/5 border-red/20 text-slate-200'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`w-2.5 h-2.5 rounded-full ${
                        seg.status === 'GOOD' ? 'bg-green shadow-sm' : 'bg-red shadow-sm'
                      }`}></div>
                      <div className="flex flex-col">
                        <span className="font-bold text-white">
                          Segment {seg.start.toFixed(2)}m – {seg.end.toFixed(2)}m
                        </span>
                        <span className="text-[9px] text-slate-400 mt-0.5">{seg.reason}</span>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <span className={`text-[9px] font-extrabold uppercase px-2 py-0.5 rounded ${
                        seg.status === 'GOOD'
                          ? 'bg-green/10 text-green border border-green/20'
                          : 'bg-red/10 text-red border border-red/20'
                      }`}>
                        {seg.status === 'GOOD' ? 'Keep (Pass)' : 'Discard (Cut)'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        ) : (
          <div className="glass-panel p-20 rounded-2xl border border-white/5 flex flex-col items-center justify-center gap-3 text-center text-slate-400 py-24 select-none">
            <Scissors className="w-10 h-10 text-gray/40" />
            <div className="text-xs uppercase tracking-widest font-mono">2D Cutting Pattern Explorer</div>
            <div className="text-[10px] max-w-sm font-sans mt-1">
              Select an inspection run from the left panel list to view its interactive flat fabric mapping and auto-generated cutting instruction guide.
            </div>
          </div>
        )}
      </div>

    </div>
  );
}

export default CuttingMapGuide;
