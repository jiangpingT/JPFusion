/**
 * PlasmaVisualization — 右侧等离子体分布可视化面板
 *
 * 包含：
 *   - Plotly.js 热图：等离子体截面温度分布（蓝→红，物理配色）
 *   - 物理量参数卡片（德拜长度 / 等离子体频率 / 阿尔文速度 / β值）
 *   - 数据来源标识（AI 模型 / 物理解析解）
 *   - 空态引导提示
 */

import React, { useEffect, useRef } from 'react';
import type { InferenceResult } from '../../api/client';

interface Props {
  result: InferenceResult | null;
}

// 科学计数法格式化
function sci(val: number, digits = 3): string {
  if (val === 0) return '0';
  const exp  = Math.floor(Math.log10(Math.abs(val)));
  const base = (val / Math.pow(10, exp)).toFixed(digits - 1);
  return `${base}×10^${exp}`;
}

type PhysicsKey = 'lambda_D' | 'omega_p' | 'v_alfven' | 'beta';

// ─── 物理卡片配置 ──────────────────────────────────────────────────────────
const PHYSICS_CARDS: Array<{
  key:   PhysicsKey;
  label: string;
  cn:    string;
  unit:  string;
  color: string;
  desc:  string;
}> = [
  {
    key:   'lambda_D',
    label: 'λ_D',
    cn:    '德拜长度',
    unit:  'm',
    color: '#38bdf8',
    desc:  '等离子体中电场屏蔽的特征距离',
  },
  {
    key:   'omega_p',
    label: 'ω_p',
    cn:    '等离子体频率',
    unit:  'rad/s',
    color: '#818cf8',
    desc:  '等离子体集体振荡的特征频率',
  },
  {
    key:   'v_alfven',
    label: 'v_A',
    cn:    '阿尔文速度',
    unit:  'm/s',
    color: '#34d399',
    desc:  '磁流体中磁扰动传播的速度',
  },
  {
    key:   'beta',
    label: 'β',
    cn:    '等离子体 β 值',
    unit:  '无量纲',
    color: '#fb923c',
    desc:  '热压与磁压之比，越小约束越好',
  },
];

// ─── 组件 ────────────────────────────────────────────────────────────────────
export function PlasmaVisualization({ result }: Props) {
  const plotRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!result || !plotRef.current) return;

    // 动态加载 Plotly（避免 bundle 过大影响初始渲染）
    import('plotly.js-dist-min').then(Plotly => {
      const T  = result.T_values;
      const gs = result.grid_size;

      // ── 极坐标 → 笛卡尔圆形截面转换 ────────────────────────────────────
      // 输出分辨率：对原始极坐标网格上采样 3 倍，保证圆形边缘光滑
      const N    = Math.max(gs * 3, 96);
      const step = 2 / (N - 1);

      const xAxis = Array.from({ length: N }, (_, i) => +(-1 + i * step).toFixed(3));
      const yAxis = Array.from({ length: N }, (_, j) => +(-1 + j * step).toFixed(3));

      // T_cart[j][i] 对应 Cartesian 点 (xAxis[i], yAxis[j])
      const T_cart:    (number | null)[][] = [];
      const psiN_cart: (number | null)[][] | null = result.psiN_profile ? [] : null;

      for (let j = 0; j < N; j++) {
        const cy = yAxis[j];
        const T_row:    (number | null)[] = [];
        const psiN_row: (number | null)[] = [];

        for (let i = 0; i < N; i++) {
          const cx = xAxis[i];
          const r  = Math.sqrt(cx * cx + cy * cy);

          if (r > 1.01) {
            // 圆外 → 透明（null 让 Plotly 不填色）
            T_row.push(null);
            psiN_row.push(null);
            continue;
          }

          // 双线性插值：用 r / θ 的小数部分做加权平均，消除像素块感
          let theta = Math.atan2(cy, cx);
          if (theta < 0) theta += 2 * Math.PI;

          const rf     = r * (gs - 1);
          const thetaf = (theta / (2 * Math.PI)) * (gs - 1);

          const ir0     = Math.min(Math.floor(rf),     gs - 2);
          const ir1     = ir0 + 1;
          const itheta0 = Math.min(Math.floor(thetaf), gs - 2);
          const itheta1 = itheta0 + 1;

          const wr1 = rf     - ir0;     const wr0 = 1 - wr1;
          const wt1 = thetaf - itheta0; const wt0 = 1 - wt1;

          const bilinear = (arr: number[][]) =>
            arr[itheta0][ir0] * wt0 * wr0 +
            arr[itheta0][ir1] * wt0 * wr1 +
            arr[itheta1][ir0] * wt1 * wr0 +
            arr[itheta1][ir1] * wt1 * wr1;

          T_row.push(bilinear(T));
          if (result.psiN_profile) {
            psiN_row.push(bilinear(result.psiN_profile));
          }
        }

        T_cart.push(T_row);
        if (psiN_cart) psiN_cart.push(psiN_row);
      }

      // ── 主热图 trace ─────────────────────────────────────────────────────
      const trace = {
        z:          T_cart,
        x:          xAxis,
        y:          yAxis,
        type:       'heatmap' as const,
        colorscale: [
          [0.0,  '#0a0a2e'],
          [0.15, '#1a1a6e'],
          [0.3,  '#0000ff'],
          [0.5,  '#00bfff'],
          [0.65, '#00ff7f'],
          [0.8,  '#ffff00'],
          [0.9,  '#ff8c00'],
          [1.0,  '#ff2400'],
        ],
        colorbar: {
          title:     'T (K)',
          titlefont: { color: '#64748b', size: 11 },
          tickfont:  { color: '#64748b', size: 9 },
          thickness: 12,
          len:       0.75,
          x:         1.02,
        },
        zmin: result.T_min,
        zmax: result.T_max,
        hovertemplate: 'x=%{x}<br>y=%{y}<br>T=%{z:.3e} K<extra></extra>',
      };

      const traces: object[] = [trace];

      // ── ψ_N 磁通面等值线（GS 模式，圆形坐标系下为同心椭圆）──────────────
      if (psiN_cart) {
        // 内层磁通面 ψ_N = 0.2 / 0.4 / 0.6 / 0.8（青绿细线）
        traces.push({
          z:          psiN_cart,
          x:          xAxis,
          y:          yAxis,
          type:       'contour',
          showscale:  false,
          colorscale: [[0, '#2dd4bf'], [1, '#2dd4bf']],
          contours: {
            start:      0.2,
            end:        0.85,
            size:       0.2,
            coloring:   'lines',
            showlabels: true,
            labelfont:  { color: '#5eead4', size: 8 },
          },
          line:      { width: 1.0 },
          opacity:   0.70,
          name:      '磁通面',
          hovertemplate: 'x=%{x}<br>y=%{y}<br>ψ_N=%{z:.2f}<extra></extra>',
        });

        // LCFS ψ_N ≈ 1.0（橙色粗线，最后封闭磁通面）
        traces.push({
          z:          psiN_cart,
          x:          xAxis,
          y:          yAxis,
          type:       'contour',
          showscale:  false,
          colorscale: [[0, '#f97316'], [1, '#f97316']],
          contours: {
            start:      0.98,
            end:        1.02,
            size:       0.04,
            coloring:   'lines',
            showlabels: true,
            labelfont:  { color: '#fb923c', size: 9 },
          },
          line:      { width: 2.5 },
          opacity:   0.92,
          name:      'LCFS',
          hovertemplate: 'LCFS ψ_N=%{z:.2f}<extra></extra>',
        });
      }

      const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor:  '#0a0f1e',
        font:          { color: '#94a3b8', size: 10 },
        margin:        { t: 20, r: 80, b: 50, l: 60 },
        showlegend:    !!psiN_cart,
        legend: {
          x: 1.08, y: 0.5,
          font:        { color: '#64748b', size: 9 },
          bgcolor:     'rgba(0,0,0,0)',
          borderwidth: 0,
        },
        // 关键：等比例坐标轴，让圆形不变成椭圆
        xaxis: {
          title:       { text: '水平位置（归一化截面坐标）', font: { color: '#64748b', size: 10 } },
          tickfont:    { color: '#475569', size: 9 },
          gridcolor:   '#1e293b',
          showgrid:    true,
          range:       [-1.15, 1.15],
          scaleanchor: 'y',
          scaleratio:  1,
        },
        yaxis: {
          title:     { text: '垂直位置（归一化截面坐标）', font: { color: '#64748b', size: 10 } },
          tickfont:  { color: '#475569', size: 9 },
          gridcolor: '#1e293b',
          showgrid:  true,
          range:     [-1.15, 1.15],
        },
        // 圆形边界虚线
        shapes: [{
          type:      'circle',
          x0: -1, y0: -1, x1: 1, y1: 1,
          xref: 'x', yref: 'y',
          line: { color: '#334155', width: 1.5, dash: 'dot' },
        }],
      };

      (Plotly as unknown as { react: Function }).react(plotRef.current!, traces, layout, {
        responsive:      true,
        displayModeBar:  true,
        displaylogo:     false,
        modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d'],
      });
    });
  }, [result]);

  // 径向温度剖面（θ=0 那一行）
  const radialProfile = result
    ? result.T_values[0].map((T: number, i: number) => ({
        r: (i / (result.grid_size - 1)).toFixed(2),
        T: T,
      }))
    : null;

  return (
    <div style={styles.panel}>
      {/* 标题 */}
      <div style={styles.header}>
        <span style={styles.title}>🌡 等离子体分布</span>
        {result && (
          <span style={{
            ...styles.sourceBadge,
            background: result.source === 'model'
              ? '#065f46'
              : result.source === 'physics_gs'
              ? '#0f4c3a'
              : '#1e3a5f',
            color: result.source === 'model'
              ? '#4ade80'
              : result.source === 'physics_gs'
              ? '#2dd4bf'
              : '#60a5fa',
          }}>
            {result.source === 'model'
              ? '🤖 AI 推理'
              : result.source === 'physics_gs'
              ? '🌀 GS 平衡'
              : '🧮 物理解析'}
          </span>
        )}
      </div>

      {/* 热图区域 */}
      <div style={styles.heatmapContainer}>
        {result ? (
          <div ref={plotRef} style={{ width: '100%', height: '100%' }} />
        ) : (
          <div style={styles.emptyState}>
            <div style={styles.emptyIcon}>⚛</div>
            <div style={styles.emptyText}>
              调节左侧参数后<br />点击「推理」按钮<br />查看等离子体温度分布
            </div>
            <div style={styles.emptyHint}>
              首次推理：无需训练，直接使用物理解析解<br />
              训练完成后：自动切换为 AI 预测模式
            </div>
          </div>
        )}
      </div>

      {/* 物理量卡片 */}
      {result && (
        <>
          <div style={styles.cardGrid}>
            {PHYSICS_CARDS.map(c => (
              <div key={c.key} style={styles.card} title={c.desc}>
                <div style={styles.cardLabel}>{c.label}</div>
                <div style={styles.cardCn}>{c.cn}</div>
                <div style={{ ...styles.cardValue, color: c.color }}>
                  {sci(result.physics_params[c.key])}
                </div>
                <div style={styles.cardUnit}>{c.unit}</div>
              </div>
            ))}
          </div>

          {/* Grad-Shafranov 平衡参数（仅 GS 求解时显示） */}
          {result.source === 'physics_gs' && result.gs_meta && (
            <div style={styles.gsBlock}>
              <div style={styles.gsTitle}>
                <span>🌀 Grad-Shafranov 平衡参数</span>
                <span style={styles.gsBadge}>FreeGS 求解</span>
              </div>
              <div style={styles.gsGrid}>
                <div style={styles.gsCard}>
                  <div style={styles.gsLabel}>I_p</div>
                  <div style={styles.gsCn}>等离子体电流</div>
                  <div style={styles.gsValue}>
                    {(result.gs_meta.plasma_current / 1e6).toFixed(2)} MA
                  </div>
                </div>
                <div style={styles.gsCard}>
                  <div style={styles.gsLabel}>β_p</div>
                  <div style={styles.gsCn}>极向 Beta</div>
                  <div style={styles.gsValue}>
                    {result.gs_meta.poloidal_beta.toFixed(3)}
                  </div>
                </div>
                <div style={styles.gsCard}>
                  <div style={styles.gsLabel}>q₉₅</div>
                  <div style={styles.gsCn}>安全因子 95%</div>
                  <div style={styles.gsValue}>
                    {result.gs_meta.q95 != null ? result.gs_meta.q95.toFixed(2) : '—'}
                  </div>
                </div>
                <div style={styles.gsCard}>
                  <div style={styles.gsLabel}>q_axis</div>
                  <div style={styles.gsCn}>轴线安全因子</div>
                  <div style={styles.gsValue}>
                    {result.gs_meta.q_axis != null ? result.gs_meta.q_axis.toFixed(2) : '—'}
                  </div>
                </div>
                <div style={styles.gsCard}>
                  <div style={styles.gsLabel}>ψ_axis</div>
                  <div style={styles.gsCn}>轴线磁通</div>
                  <div style={styles.gsValue}>
                    {result.gs_meta.psi_axis.toFixed(4)} Wb
                  </div>
                </div>
                <div style={styles.gsCard}>
                  <div style={styles.gsLabel}>R₀ / a</div>
                  <div style={styles.gsCn}>大/小半径</div>
                  <div style={styles.gsValue}>
                    {result.gs_meta.R0.toFixed(2)} / {result.gs_meta.a.toFixed(2)} m
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* MC Dropout 不确定性信息 */}
          {result.T_uncertainty && (
            <div style={styles.uncertaintyBlock}>
              <div style={styles.uncertaintyTitle}>
                <span>🎲 MC Dropout 不确定性估计</span>
                <span style={styles.uncertaintyBadge}>N=20 采样</span>
              </div>
              <div style={styles.uncertaintyRow}>
                <div style={styles.uncertaintyCard}>
                  <div style={styles.ucLabel}>σ_mean</div>
                  <div style={styles.ucCn}>平均标准差</div>
                  <div style={styles.ucValue}>
                    {sci(result.T_uncertainty_mean ?? 0)} K
                  </div>
                </div>
                <div style={styles.uncertaintyCard}>
                  <div style={styles.ucLabel}>σ_max</div>
                  <div style={styles.ucCn}>最大标准差</div>
                  <div style={styles.ucValue}>
                    {sci(result.T_uncertainty_max ?? 0)} K
                  </div>
                </div>
                <div style={styles.uncertaintyCard}>
                  <div style={styles.ucLabel}>σ/T</div>
                  <div style={styles.ucCn}>相对不确定性</div>
                  <div style={styles.ucValue}>
                    {result.T_max > 0
                      ? ((result.T_uncertainty_mean ?? 0) / result.T_max * 100).toFixed(1) + '%'
                      : '—'}
                  </div>
                </div>
              </div>
              {/* 不确定性径向条形图 */}
              <div style={styles.ucProfileTitle}>不确定性径向分布（θ=0°）</div>
              <div style={styles.profileChart}>
                {result.T_uncertainty[0].map((sigma: number, i: number) => {
                  const maxSig = Math.max(...result.T_uncertainty![0]);
                  const height = maxSig > 0 ? (sigma / maxSig) * 100 : 0;
                  return (
                    <div
                      key={i}
                      title={`r=${(i / (result.grid_size - 1)).toFixed(2)}, σ=${sigma.toExponential(2)} K`}
                      style={{
                        flex:         1,
                        height:       `${height}%`,
                        background:   `hsla(${280 - height * 0.5}, 70%, 60%, 0.8)`,
                        alignSelf:    'flex-end',
                        minWidth:     '2px',
                        borderRadius: '1px 1px 0 0',
                      }}
                    />
                  );
                })}
              </div>
              <div style={styles.profileAxis}>
                <span>中心 r=0</span>
                <span>边界 r=1</span>
              </div>
            </div>
          )}

          {/* 径向温度剖面 */}
          {radialProfile && (
            <div style={styles.profileBlock}>
              <div style={styles.profileTitle}>径向温度剖面（θ = 0°）</div>
              <div style={styles.profileChart}>
                {radialProfile.map((pt, i) => {
                  const maxT   = Math.max(...radialProfile.map(p => p.T));
                  const height = maxT > 0 ? (pt.T / maxT) * 100 : 0;
                  const hue    = 240 - (height / 100) * 240;  // 蓝→红
                  return (
                    <div
                      key={i}
                      title={`r=${pt.r}, T=${pt.T.toExponential(2)} K`}
                      style={{
                        flex:             1,
                        height:           `${height}%`,
                        background:       `hsl(${hue}, 90%, 55%)`,
                        alignSelf:        'flex-end',
                        minWidth:         '2px',
                        borderRadius:     '1px 1px 0 0',
                        transition:       'height 0.3s ease',
                      }}
                    />
                  );
                })}
              </div>
              <div style={styles.profileAxis}>
                <span>中心 r=0</span>
                <span>边界 r=1</span>
              </div>
            </div>
          )}

          {/* 名词备注 */}
          <div style={styles.glossary}>
            <div style={styles.glossaryTitle}>【名词备注】</div>
            {PHYSICS_CARDS.map(c => (
              <div key={c.key} style={styles.glossaryItem}>
                <b>{c.label}（{c.cn}）</b> — {c.desc}
              </div>
            ))}
            <div style={styles.glossaryItem}>
              <b>SOL（刮削层）</b> — Scrape-Off Layer，LCFS（最后封闭磁通面）外侧区域，温度以特征长度 λ_q≈0.03 指数衰减，比原来的高斯截断更真实
            </div>
            {result?.T_uncertainty && (
              <div style={styles.glossaryItem}>
                <b>MC Dropout σ</b> — Monte Carlo Dropout 不确定性，模型对某点预测的置信度指标；σ 越大说明该区域数据稀疏或物理边界复杂，AI 在此处把握较低
              </div>
            )}
            {result?.source === 'physics_gs' && (
              <>
                <div style={styles.glossaryItem}>
                  <b>GS 方程（Grad-Shafranov）</b> — Δ*ψ = -μ₀R²dp/dψ - F·dF/dψ，描述托卡马克磁平衡的核心偏微分方程，求解得到磁通面 ψ_N 真实坐标
                </div>
                <div style={styles.glossaryItem}>
                  <b>ψ_N（归一化极向磁通）</b> — 0=磁轴，1=LCFS（最后封闭磁通面），温度剖面 T(ψ_N) 比简单径向坐标物理精度高一个量级
                </div>
                <div style={styles.glossaryItem}>
                  <b>q₉₅（安全因子）</b> — 磁力线绕大环一圈对应绕小环的圈数，q₉₅ 在 95% 磁通面处，物理约束 q₉₅≥2 可防止磁不稳定性
                </div>
                <div style={styles.glossaryItem}>
                  <b>β_p（极向 Beta）</b> — 热压与极向磁压之比，β_p 反映等离子体约束效率，ITER 设计值约 0.8-1.0
                </div>
              </>
            )}
          </div>
        </>
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
    background: '#050b18',
    padding: '20px 24px',
    color: '#e2e8f0',
    fontFamily: "'Inter', 'SF Pro Display', sans-serif",
    fontSize: '13px',
    boxSizing: 'border-box',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '12px',
  },
  title: {
    fontSize: '16px',
    fontWeight: 700,
    color: '#f97316',
  },
  sourceBadge: {
    padding: '3px 8px',
    borderRadius: '4px',
    fontSize: '11px',
    fontWeight: 600,
  },
  heatmapContainer: {
    height: '500px',
    background: '#0a0f1e',
    borderRadius: '8px',
    marginBottom: '16px',
    overflow: 'hidden',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  emptyState: {
    textAlign: 'center',
    padding: '20px',
  },
  emptyIcon: {
    fontSize: '40px',
    marginBottom: '12px',
    opacity: 0.3,
  },
  emptyText: {
    color: '#475569',
    lineHeight: '1.8',
    marginBottom: '12px',
  },
  emptyHint: {
    fontSize: '11px',
    color: '#334155',
    lineHeight: '1.6',
  },
  cardGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap: '8px',
    marginBottom: '12px',
  },
  card: {
    background: '#0f172a',
    border: '1px solid #1e293b',
    borderRadius: '8px',
    padding: '10px',
    cursor: 'help',
  },
  cardLabel: {
    fontSize: '14px',
    fontFamily: 'monospace',
    color: '#94a3b8',
    fontWeight: 600,
  },
  cardCn: {
    fontSize: '10px',
    color: '#475569',
    marginBottom: '4px',
  },
  cardValue: {
    fontSize: '13px',
    fontFamily: 'monospace',
    fontWeight: 700,
  },
  cardUnit: {
    fontSize: '10px',
    color: '#334155',
    marginTop: '2px',
  },
  uncertaintyBlock: {
    background: '#0f172a',
    border: '1px solid #312e81',
    borderRadius: '8px',
    padding: '12px',
    marginBottom: '12px',
  },
  uncertaintyTitle: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    fontSize: '11px',
    fontWeight: 600,
    color: '#a78bfa',
    marginBottom: '8px',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
  },
  uncertaintyBadge: {
    fontSize: '9px',
    padding: '2px 6px',
    background: '#312e81',
    color: '#c4b5fd',
    borderRadius: '4px',
  },
  uncertaintyRow: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '8px',
    marginBottom: '10px',
  },
  uncertaintyCard: {
    background: '#1e293b',
    borderRadius: '6px',
    padding: '8px',
    textAlign: 'center' as const,
  },
  ucLabel: {
    fontSize: '12px',
    fontFamily: 'monospace',
    color: '#94a3b8',
    fontWeight: 600,
  },
  ucCn: {
    fontSize: '9px',
    color: '#475569',
    marginBottom: '4px',
  },
  ucValue: {
    fontSize: '11px',
    fontFamily: 'monospace',
    color: '#a78bfa',
    fontWeight: 700,
  },
  ucProfileTitle: {
    fontSize: '10px',
    color: '#475569',
    marginBottom: '5px',
  },
  profileBlock: {
    background: '#0f172a',
    borderRadius: '8px',
    padding: '12px',
    marginBottom: '12px',
  },
  profileTitle: {
    fontSize: '11px',
    color: '#475569',
    marginBottom: '8px',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  profileChart: {
    height: '80px',
    display: 'flex',
    alignItems: 'flex-end',
    gap: '1px',
    background: '#0a0f1e',
    padding: '4px',
    borderRadius: '4px',
  },
  profileAxis: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '9px',
    color: '#334155',
    marginTop: '4px',
  },
  glossary: {
    padding: '12px',
    background: '#0f172a',
    borderRadius: '8px',
    borderLeft: '3px solid #f97316',
  },
  glossaryTitle: {
    fontSize: '11px',
    fontWeight: 700,
    color: '#f97316',
    marginBottom: '8px',
  },
  glossaryItem: {
    fontSize: '10px',
    color: '#64748b',
    marginBottom: '4px',
    lineHeight: '1.5',
  },
  gsBlock: {
    background: '#0f172a',
    border: '1px solid #0f4c3a',
    borderRadius: '8px',
    padding: '12px',
    marginBottom: '12px',
  },
  gsTitle: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    fontSize: '11px',
    fontWeight: 600,
    color: '#2dd4bf',
    marginBottom: '8px',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
  },
  gsBadge: {
    fontSize: '9px',
    padding: '2px 6px',
    background: '#0f4c3a',
    color: '#5eead4',
    borderRadius: '4px',
  },
  gsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '8px',
  },
  gsCard: {
    background: '#1e293b',
    borderRadius: '6px',
    padding: '8px',
    textAlign: 'center' as const,
  },
  gsLabel: {
    fontSize: '12px',
    fontFamily: 'monospace',
    color: '#94a3b8',
    fontWeight: 600,
  },
  gsCn: {
    fontSize: '9px',
    color: '#475569',
    marginBottom: '4px',
  },
  gsValue: {
    fontSize: '12px',
    fontFamily: 'monospace',
    color: '#2dd4bf',
    fontWeight: 700,
  },
};
