import React, { useState, useEffect } from 'react';
import { Camera, Download, RefreshCw, CheckCircle2, AlertCircle } from 'lucide-react';

function SampleCollector() {
  const [samples, setSamples] = useState([]);
  const [capturing, setCapturing] = useState(false);
  const [statusMsg, setStatusMsg] = useState(null);

  // Load recently captured sample images
  const loadSamples = () => {
    fetch('/api/samples')
      .then(res => res.json())
      .then(data => {
        setSamples(data || []);
      })
      .catch(err => console.error("Failed to load sample gallery", err));
  };

  useEffect(() => {
    loadSamples();
  }, []);

  // Trigger camera frame capture
  const handleCapture = () => {
    setCapturing(true);
    setStatusMsg(null);
    fetch('/api/samples/capture', { method: 'POST' })
      .then(res => res.json())
      .then(data => {
        setCapturing(false);
        if (data.status === 'success') {
          setStatusMsg({ type: 'success', text: `Sample captured successfully as ${data.filename}!` });
          loadSamples();
          // Clear status after 3s
          setTimeout(() => setStatusMsg(null), 4000);
        } else {
          setStatusMsg({ type: 'error', text: data.message || "Failed to capture sample" });
        }
      })
      .catch(err => {
        setCapturing(false);
        setStatusMsg({ type: 'error', text: `Network error: ${err.message}` });
      });
  };

  return (
    <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start text-white max-w-7xl mx-auto">
      
      {/* LEFT COLUMN: Live Camera View & Capture Button (Col-span 8) */}
      <div className="xl:col-span-8 flex flex-col gap-6">
        <div className="glass-panel p-5 rounded-2xl border border-white/5 flex flex-col gap-4 relative overflow-hidden">
          <h3 className="text-xs font-bold text-gray uppercase tracking-widest flex items-center gap-1.5 border-b border-white/5 pb-2">
            <Camera className="w-4 h-4 text-cyan" />
            <span>Sample Collection Live View</span>
          </h3>

          {/* Camera feed canvas wrapper */}
          <div className="relative w-full rounded-2xl border border-white/10 bg-black overflow-hidden flex items-center justify-center min-h-[350px] md:min-h-[460px] shadow-inner shadow-black/80">
            {/* Live blinking tag */}
            <div className="absolute top-4 left-4 z-10 bg-cyan/15 border border-cyan/30 text-cyan text-[9px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider flex items-center gap-1 live-blink">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan"></span> Camera 1 Feed
            </div>
            
            <img
              src="/video_feed/1"
              alt="Live Fabric Feed"
              className="w-full h-full object-cover max-h-[500px]"
              onError={(e) => {
                e.target.src = "https://images.unsplash.com/photo-1582738411706-bfc8e691d1c2?q=80&w=600&auto=format&fit=crop";
              }}
            />
          </div>

          {/* Action Capture Bar */}
          <div className="flex flex-col md:flex-row justify-between items-center gap-4 bg-navy/20 p-4 rounded-xl border border-white/5">
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] text-gray uppercase font-bold">Fabric Training Collector</span>
              <span className="text-[9px] text-slate-400 font-sans">
                Snapshots are stored in <code className="bg-white/5 px-1 rounded">saved_frames/samples/</code> folder as raw dataset files.
              </span>
            </div>

            <button
              onClick={handleCapture}
              disabled={capturing}
              className={`px-6 py-3.5 bg-blue-600 hover:bg-blue-700 text-white font-extrabold text-xs uppercase tracking-wider rounded-xl shadow-lg shadow-blue-600/20 transition-all flex items-center gap-2 cursor-pointer hover:scale-103 active:scale-97 disabled:opacity-50`}
            >
              {capturing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" /> Capturing...
                </>
              ) : (
                <>
                  <Camera className="w-4 h-4" /> Capture Sample Frame
                </>
              )}
            </button>
          </div>

          {/* Status Message Overlay */}
          {statusMsg && (
            <div className={`mt-2 p-3.5 rounded-xl border flex items-center gap-2 text-xs font-mono justify-center animate-fade-in ${
              statusMsg.type === 'success' 
                ? 'bg-green/10 border-green/25 text-green' 
                : 'bg-red/10 border-red/25 text-red'
            }`}>
              {statusMsg.type === 'success' ? <CheckCircle2 className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
              <span>{statusMsg.text}</span>
            </div>
          )}
        </div>
      </div>

      {/* RIGHT COLUMN: Captured Gallery (Col-span 4) */}
      <div className="xl:col-span-4 glass-panel p-5 rounded-2xl border border-white/5 flex flex-col gap-4">
        <div className="flex justify-between items-center border-b border-white/5 pb-2">
          <h3 className="text-xs font-bold text-gray uppercase tracking-widest flex items-center gap-1.5">
            <Camera className="w-4 h-4 text-cyan" />
            <span>Captured Samples ({samples.length})</span>
          </h3>
          <button 
            onClick={loadSamples}
            title="Reload gallery"
            className="text-gray hover:text-white p-1 hover:bg-white/5 rounded transition cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>

        {samples.length === 0 ? (
          <div className="text-[10px] text-gray uppercase tracking-wider font-mono text-center py-20">
            No training samples captured yet.
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3 max-h-[460px] overflow-y-auto pr-1">
            {samples.map((filename) => (
              <div 
                key={filename} 
                className="group relative bg-slate-900 border border-white/5 rounded-xl overflow-hidden aspect-square flex items-center justify-center shadow-md hover:border-cyan/40 transition-all"
              >
                <img
                  src={`/samples/${filename}`}
                  alt={filename}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                />
                
                {/* Filename hover label */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity p-2 flex flex-col justify-end text-[8px] font-mono text-slate-300">
                  <span className="truncate">{filename}</span>
                  <a
                    href={`/samples/${filename}`}
                    download
                    className="mt-1 text-cyan hover:underline flex items-center gap-0.5 text-[7px] uppercase font-sans font-bold"
                  >
                    <Download className="w-2 h-2" /> Download
                  </a>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

    </div>
  );
}

export default SampleCollector;
