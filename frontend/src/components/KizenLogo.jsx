import React from 'react';

function KizenLogo({ className = "h-10 w-auto", lightText = true, gearOnly = false }) {
  if (gearOnly) {
    return (
      <svg className={className} viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="18" cy="18" r="15" stroke="#00B8D4" strokeWidth="2.5" strokeDasharray="4 2" />
        <circle cx="18" cy="18" r="8" fill="#2563EB" fillOpacity="0.2" stroke="#2563EB" strokeWidth="2" />
        <path d="M18 6V10M18 26V30M6 18H10M26 18H30" stroke="#00B8D4" strokeWidth="2.5" strokeLinecap="round" />
        <circle cx="18" cy="18" r="3.5" fill="#00B8D4" />
      </svg>
    );
  }

  return (
    <a 
      href="https://kizen.co.in/" 
      target="_blank" 
      rel="noopener noreferrer"
      className="flex items-center gap-3 no-underline group transition-transform duration-200 hover:scale-[1.02]"
      title="Kizen Engineering — Innovation Is Our Tradition (kizen.co.in)"
    >
      <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500/20 via-blue-600/20 to-cyan-400/10 border border-cyan/30 flex items-center justify-center shadow-md shadow-cyan/10 group-hover:border-cyan/60 transition-colors">
        <svg className="w-6 h-6" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="18" cy="18" r="14" stroke="#00B8D4" strokeWidth="2.2" strokeDasharray="3 2" />
          <circle cx="18" cy="18" r="7" fill="#2563EB" fillOpacity="0.3" stroke="#2563EB" strokeWidth="2" />
          <path d="M18 5V9M18 27V31M5 18H9M27 18H31" stroke="#00B8D4" strokeWidth="2.2" strokeLinecap="round" />
          <circle cx="18" cy="18" r="3" fill="#00B8D4" />
        </svg>
      </div>
      <div className="flex flex-col">
        <div className="flex items-center gap-1.5 font-black tracking-[0.08em] font-sans uppercase text-sm leading-tight">
          <span className={lightText ? "text-white" : "text-slate-900"}>KIZEN</span>
          <span className="text-cyan font-extrabold">ENGINEERING</span>
        </div>
        <span className="text-[8.5px] font-bold tracking-[0.18em] uppercase text-cyan/90 font-mono mt-0.5">
          Innovation Is Our Tradition
        </span>
      </div>
    </a>
  );
}

export default KizenLogo;
