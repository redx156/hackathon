'use client';
import { useRef, useEffect, useCallback, useMemo } from 'react';
import { gsap } from 'gsap';
import './DotGrid.css';

const throttle = (func, limit) => {
    let lastCall = 0;
    return function (...args) {
        const now = performance.now();
        if (now - lastCall >= limit) {
            lastCall = now;
            func.apply(this, args);
        }
    };
};

function hexToRgb(hex) {
    const m = hex.match(/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i);
    if (!m) return { r: 0, g: 0, b: 0 };
    return { r: parseInt(m[1], 16), g: parseInt(m[2], 16), b: parseInt(m[3], 16) };
}

const DotGrid = ({ dotSize = 16, gap = 32, baseColor = '#5227FF', activeColor = '#5227FF', proximity = 150, shockRadius = 250, shockStrength = 5, returnDuration = 1.5, className = '', style }) => {
    const wrapperRef = useRef(null);
    const canvasRef = useRef(null);
    const dotsRef = useRef([]);
    const pointerRef = useRef({ x: 0, y: 0 });

    const baseRgb = useMemo(() => hexToRgb(baseColor), [baseColor]);
    const activeRgb = useMemo(() => hexToRgb(activeColor), [activeColor]);

    const circlePath = useMemo(() => {
        if (typeof window === 'undefined' || !window.Path2D) return null;
        const p = new window.Path2D();
        p.arc(0, 0, dotSize / 2, 0, Math.PI * 2);
        return p;
    }, [dotSize]);

    const buildGrid = useCallback(() => {
        const wrap = wrapperRef.current;
        const canvas = canvasRef.current;
        if (!wrap || !canvas) return;
        const { width, height } = wrap.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
        const ctx = canvas.getContext('2d');
        if (ctx) ctx.scale(dpr, dpr);
        const cols = Math.floor((width + gap) / (dotSize + gap));
        const rows = Math.floor((height + gap) / (dotSize + gap));
        const cell = dotSize + gap;
        const gridW = cell * cols - gap;
        const gridH = cell * rows - gap;
        const startX = (width - gridW) / 2 + dotSize / 2;
        const startY = (height - gridH) / 2 + dotSize / 2;
        const dots = [];
        for (let y = 0; y < rows; y++) {
            for (let x = 0; x < cols; x++) {
                dots.push({ cx: startX + x * cell, cy: startY + y * cell, xOffset: 0, yOffset: 0, _inertiaApplied: false });
            }
        }
        dotsRef.current = dots;
    }, [dotSize, gap]);

    useEffect(() => {
        if (!circlePath) return;
        let rafId;
        const proxSq = proximity * proximity;
        const draw = () => {
            const canvas = canvasRef.current;
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const { x: px, y: py } = pointerRef.current;
            for (const dot of dotsRef.current) {
                const ox = dot.cx + dot.xOffset;
                const oy = dot.cy + dot.yOffset;
                const dx = dot.cx - px;
                const dy = dot.cy - py;
                const dsq = dx * dx + dy * dy;
                let style = baseColor;
                if (dsq <= proxSq) {
                    const t = 1 - Math.sqrt(dsq) / proximity;
                    const r = Math.round(baseRgb.r + (activeRgb.r - baseRgb.r) * t);
                    const g = Math.round(baseRgb.g + (activeRgb.g - baseRgb.g) * t);
                    const b = Math.round(baseRgb.b + (activeRgb.b - baseRgb.b) * t);
                    style = `rgb(${r},${g},${b})`;
                }
                ctx.save();
                ctx.translate(ox, oy);
                ctx.fillStyle = style;
                ctx.fill(circlePath);
                ctx.restore();
            }
            rafId = requestAnimationFrame(draw);
        };
        draw();
        return () => cancelAnimationFrame(rafId);
    }, [proximity, baseColor, activeRgb, baseRgb, circlePath]);

    useEffect(() => {
        buildGrid();
        let ro = null;
        if ('ResizeObserver' in window) {
            ro = new ResizeObserver(buildGrid);
            wrapperRef.current && ro.observe(wrapperRef.current);
        } else {
            window.addEventListener('resize', buildGrid);
        }
        return () => { if (ro) ro.disconnect(); else window.removeEventListener('resize', buildGrid); };
    }, [buildGrid]);

    useEffect(() => {
        const onMove = e => {
            const rect = canvasRef.current?.getBoundingClientRect();
            if (!rect) return;
            pointerRef.current.x = e.clientX - rect.left;
            pointerRef.current.y = e.clientY - rect.top;
        };
        const onClick = e => {
            const rect = canvasRef.current?.getBoundingClientRect();
            if (!rect) return;
            const cx = e.clientX - rect.left;
            const cy = e.clientY - rect.top;
            for (const dot of dotsRef.current) {
                const dist = Math.hypot(dot.cx - cx, dot.cy - cy);
                if (dist < shockRadius && !dot._inertiaApplied) {
                    dot._inertiaApplied = true;
                    gsap.killTweensOf(dot);
                    const falloff = Math.max(0, 1 - dist / shockRadius);
                    gsap.to(dot, {
                        xOffset: (dot.cx - cx) * shockStrength * falloff,
                        yOffset: (dot.cy - cy) * shockStrength * falloff,
                        duration: 0.3,
                        onComplete: () => {
                            gsap.to(dot, { xOffset: 0, yOffset: 0, duration: returnDuration, ease: 'elastic.out(1,0.75)' });
                            dot._inertiaApplied = false;
                        }
                    });
                }
            }
        };
        const throttledMove = throttle(onMove, 50);
        window.addEventListener('mousemove', throttledMove, { passive: true });
        window.addEventListener('click', onClick);
        return () => { window.removeEventListener('mousemove', throttledMove); window.removeEventListener('click', onClick); };
    }, [shockRadius, shockStrength, returnDuration]);

    return (<section className={`dot-grid ${className}`} style={style}><div ref={wrapperRef} className="dot-grid__wrap"><canvas ref={canvasRef} className="dot-grid__canvas" /></div></section>);
};

export default DotGrid;
