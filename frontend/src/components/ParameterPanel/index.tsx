/**
 * ParameterPanel — 左侧参数配置面板
 *
 * 包含：
 *   - 模型状态 badge（内置轮询，30s interval）
 *   - 三个物理参数滑块：n_e（电子密度）/ T_e（电子温度）/ B（磁场强度）
 *   - 推理网格选择
 *   - [推理] 大按钮 + loading 状态
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { fusionApi } from '../../api/client';

// ─── 参数范围配置 ────────────────────────────────────────────────────────────
const PARAMS_CONFIG = [
  {
    key:    'n_e_exp',
    label:  'n_e 电子密度',
    unit:   'm⁻³',
    min:    18,
    max:    20,
    step:   0.1,
    toDisplay: (v: number) => `1×10^${v.toFixed(1)}`,
    toPhysical: (v: number) => Math.pow(10, v),
    hint:   '18=1e18（低密度）→ 20=1e20（高密度）',
  },
  {
    key:    'T_e_exp',
    label:  'T_e 电子温度',
    unit:   'K',
    min:    6,
    max:    8,
    step:   0.1,
    toDisplay: (v: number) => `1×10^${v.toFixed(1)}`,
    toPhysical: (v: number) => Math.pow(10, v),
    hint:   '6=1e6 K (86 eV) → 8=1e8 K (8600 eV)',
  },
  {
    key:    'B',
    label:  'B 磁场强度',
    unit:   'T',
    min:    1,
    max:    10,
    step:   0.5,
    toDisplay: (v: number) => `${v.toFixed(1)}`,
    toPhysical: (v: number) => v,
    hint:   '1T（弱约束）→ 10T（强约束，ITER 级别）',
  },
];

// ─── 组件 ────────────────────────────────────────────────────────────────────

interface Props {
  onInferenceResult: (result: unknown) => void;
}

export function ParameterPanel({ onInferenceResult }: Props) {
  const [sliders, setSliders] = useState({
    n_e_exp: 19,
    T_e_exp: 7,
    B:       5.0,
  });
  const [gridSize,     setGridSize]     = useState(32);
  const [status,       setStatus]       = useState('');
  const [loading,      setLoading]      = useState(false);
  const [modelStatus,  setModelStatus]  = useState('idle');

  const slidersRef  = useRef(sliders);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // 内置轮询模型状态（30s interval）
  useEffect(() => {
    const poll = async () => {
      try {
        const st = await fusionApi.getModelStatus();
        setModelStatus(st.training_status);
      } catch {
        // 后端未启动，忽略
      }
    };
    poll();
    const timer = setInterval(poll, 30000);
    return () => clearInterval(timer);
  }, []);

  // ── 核心推理函数（接收显式参数，避免 closure 捕获过期值）────────────────
  const runInference = useCallback(async (
    params: typeof sliders,
    grid:   number,
  ) => {
    setLoading(true);
    setStatus('推理中...');
    try {
      const result = await fusionApi.inference({
        n_e:       Math.pow(10, params.n_e_exp),
        T_e:       Math.pow(10, params.T_e_exp),
        B:         params.B,
        grid_size: grid,
      });
      onInferenceResult(result);
      setStatus(`✓ 推理完成（来源：${result.source === 'model' ? 'AI 模型' : '物理解析解'}）`);
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } }; message?: string };
      setStatus(`✗ 推理失败：${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  }, [onInferenceResult]);

  // ── 滑块变化：更新状态 + 800ms 防抖自动推理 ───────────────────────────
  const handleSlider = (key: string, value: number) => {
    const next = { ...slidersRef.current, [key]: value };
    slidersRef.current = next;
    setSliders(next);

    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      runInference(next, gridSize);
    }, 800);
  };

  // ── 手动推理按钮：立即触发，取消待执行的防抖 ─────────────────────────
  const handleInference = () => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    runInference(slidersRef.current, gridSize);
  };

  const statusColor = modelStatus === 'training'   ? '#facc15'
    : modelStatus === 'completed'  ? '#4ade80'
    : modelStatus === 'error'      ? '#f87171'
    : '#94a3b8';

  const statusText = modelStatus === 'training'    ? '训练中'
    : modelStatus === 'completed'  ? '模型就绪'
    : modelStatus === 'idle'       ? '待训练'
    : modelStatus === 'generating' ? '生成数据中'
    : modelStatus === 'error'      ? '异常'
    : '未知';

  return (
    <div style={styles.panel}>
      {/* 标题 */}
      <div style={styles.title}>⚛ 参数配置</div>

      {/* 模型状态指示 */}
      <div style={styles.statusBadge}>
        <span style={{ ...styles.dot, background: statusColor }} />
        <span style={styles.statusText}>{statusText}</span>
        <span style={styles.statusMode}>
          {modelStatus === 'completed' ? 'AI 模式' : '物理解析模式'}
        </span>
      </div>

      {/* 物理参数滑块 */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>等离子体参数</div>
        {PARAMS_CONFIG.map(cfg => {
          const val = sliders[cfg.key as keyof typeof sliders];
          return (
            <div key={cfg.key} style={styles.sliderBlock}>
              <div style={styles.sliderHeader}>
                <span style={styles.sliderLabel}>{cfg.label}</span>
                <span style={styles.sliderValue}>
                  {cfg.toDisplay(val)} <span style={styles.unit}>{cfg.unit}</span>
                </span>
              </div>
              <input
                type="range"
                min={cfg.min}
                max={cfg.max}
                step={cfg.step}
                value={val}
                style={styles.slider}
                onChange={e => handleSlider(cfg.key, parseFloat(e.target.value))}
              />
              <div style={styles.hint}>{cfg.hint}</div>
            </div>
          );
        })}
      </div>

      {/* 推理设置 */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>推理设置</div>
        <div style={styles.row}>
          <label style={styles.label}>网格分辨率</label>
          <select
            value={gridSize}
            onChange={e => setGridSize(Number(e.target.value))}
            style={styles.select}
          >
            {[16, 32, 48, 64].map(v => (
              <option key={v} value={v}>{v}×{v}</option>
            ))}
          </select>
        </div>
        <button
          style={{ ...styles.btn, ...styles.btnPrimary, ...(loading ? styles.btnDisabled : {}) }}
          onClick={handleInference}
          disabled={loading}
        >
          {loading ? '⏳ 推理中...' : '▶ 推理'}
        </button>
      </div>

      {/* 状态反馈 */}
      {status && (
        <div style={styles.feedback}>{status}</div>
      )}
    </div>
  );
}

// ─── 样式 ────────────────────────────────────────────────────────────────────
const styles: Record<string, React.CSSProperties> = {
  panel: {
    width: '100%',
    height: '100%',
    overflowY: 'auto',
    background: '#0f172a',
    padding: '16px',
    color: '#e2e8f0',
    fontFamily: "'Inter', 'SF Pro Display', sans-serif",
    fontSize: '13px',
    boxSizing: 'border-box',
  },
  title: {
    fontSize: '16px',
    fontWeight: 700,
    color: '#38bdf8',
    marginBottom: '12px',
    letterSpacing: '0.5px',
  },
  statusBadge: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    padding: '6px 10px',
    background: '#1e293b',
    borderRadius: '6px',
    marginBottom: '16px',
  },
  dot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    flexShrink: 0,
  },
  statusText: {
    fontSize: '12px',
    color: '#94a3b8',
    flex: 1,
  },
  statusMode: {
    fontSize: '10px',
    color: '#475569',
    fontStyle: 'italic',
  },
  section: {
    marginBottom: '16px',
    padding: '12px',
    background: '#1e293b',
    borderRadius: '8px',
  },
  sectionTitle: {
    fontSize: '11px',
    fontWeight: 600,
    color: '#64748b',
    textTransform: 'uppercase',
    letterSpacing: '0.8px',
    marginBottom: '10px',
  },
  sliderBlock: {
    marginBottom: '12px',
  },
  sliderHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'baseline',
    marginBottom: '4px',
  },
  sliderLabel: {
    color: '#cbd5e1',
    fontSize: '12px',
  },
  sliderValue: {
    color: '#38bdf8',
    fontWeight: 600,
    fontSize: '13px',
    fontFamily: 'monospace',
  },
  unit: {
    fontSize: '10px',
    color: '#64748b',
  },
  slider: {
    width: '100%',
    accentColor: '#38bdf8',
    cursor: 'pointer',
  },
  hint: {
    fontSize: '10px',
    color: '#475569',
    marginTop: '2px',
  },
  row: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '10px',
  },
  label: {
    color: '#94a3b8',
    fontSize: '12px',
  },
  select: {
    padding: '4px 6px',
    background: '#0f172a',
    border: '1px solid #334155',
    borderRadius: '4px',
    color: '#e2e8f0',
    fontSize: '12px',
  },
  btn: {
    width: '100%',
    padding: '10px',
    borderRadius: '6px',
    border: 'none',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 700,
    transition: 'opacity 0.2s',
  },
  btnPrimary: {
    background: 'linear-gradient(135deg, #0ea5e9, #6366f1)',
    color: '#fff',
  },
  btnDisabled: {
    opacity: 0.6,
    cursor: 'not-allowed',
  },
  feedback: {
    padding: '10px',
    background: '#1e293b',
    borderRadius: '6px',
    color: '#94a3b8',
    fontSize: '11px',
    lineHeight: '1.5',
    borderLeft: '3px solid #38bdf8',
  },
};
