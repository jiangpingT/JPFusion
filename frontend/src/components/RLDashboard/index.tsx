/**
 * RLDashboard — FusionRL 训练监控 + 四阶段推理面板
 *
 * 四阶段命名体系（数据来源 × 学习方式）：
 *   Phase 1 · Sim-RL      仿真环境在线强化学习（PPO）：v1~v6 版本对比表 + 推理
 *   Phase 2 · Sim-SFT     仿真环境监督微调（BC）：行为克隆预训练状态 + 推理
 *   Phase 3 · Offline-RL  PPO 专家轨迹 + Phase 4b 回流轨迹（数据飞轮）
 *   Phase 4 · Model-RL    世界模型强化学习（MBRL）：Dyna 循环 + 精调
 *
 * 布局：
 *   左（340px）— 训练状态监控（status / metrics / 奖励曲线 / 控制）
 *   右（flex）  — 四阶段推理面板
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';

const BASE_URL = window.location.pathname.startsWith('/jpfusion') ? '/jpfusion' : '';

// ─── 类型 ─────────────────────────────────────────────────────────────────────
interface RLStatus {
  status:          'idle' | 'training' | 'done' | 'error';
  total_steps:     number;
  mean_reward:     number | null;
  mean_ep_length:  number | null;
  disruption_rate: number | null;
  lawson_rate:     number | null;
  elapsed_s:       number;
  error_msg:       string | null;
}

interface HistoryRecord {
  timestep:        number;
  mean_reward:     number;
  mean_ep_length:  number;
  disruption_rate: number;
  lawson_rate:     number;
  gaming_proxy:    number;
}

interface ModelMeta {
  version:      string;
  final_path:   string | null;
  checkpoints:  string[];
  best_reward:  number | null;
  last_reward:  number | null;
  lawson_rate:  number | null;
  n_records:    number;
  latest_ckpt:  string | null;
}

interface PhaseStatus {
  phase1: {
    ready: boolean; models: ModelMeta[];
    best_model: string | null; best_reward: number | null;
    best_lawson: number | null; n_versions: number;
  };
  phase2: {
    sft_ready: boolean; sft_path: string | null; sft_size_mb: number | null;
    ppo_warmstart_ready: boolean; ppo_warmstart_path: string | null;
    ppo_models: ModelMeta[];
    demo_model: string | null;
    best_reward: number | null; best_lawson: number | null;
  };
  phase3: { ready: boolean; cql_path: string | null; needs_east: boolean };
  phase4: {
    world_model_ready: boolean; world_model_path: string | null;
    world_model_size_mb: number | null;
    mbrl_ready: boolean; mbrl_path: string | null;
    best_reward: number | null; best_std: number | null;
    best_lawson: number | null; disrupt_rate: number | null;
    version: string | null; notes: string | null;
  };
}

interface InferResult {
  trajectory:      Record<string, number[]>;
  total_reward:    number;
  n_steps:         number;
  disrupted:       boolean;
  lawson_achieved: boolean;
  final_lawson:    number;
  version?:        string;
  source:          string;
  lawson_target:   number;
}

// ─── 颜色 ─────────────────────────────────────────────────────────────────────
const C = {
  bg:      '#020817',
  panel:   '#0a0f1e',
  panel2:  '#0d1425',
  border:  '#1e293b',
  text:    '#e2e8f0',
  muted:   '#64748b',
  dim:     '#334155',
  accent:  '#38bdf8',
  success: '#10b981',
  warning: '#f59e0b',
  danger:  '#ef4444',
  purple:  '#818cf8',
  green:   '#34d399',
  // Phase 专属色（命名：Sim-RL / Sim-SFT / Offline-RL / Model-RL）
  p1: '#38bdf8',   // 蓝   — Sim-RL    (PPO)
  p2: '#f59e0b',   // 琥珀 — Sim-SFT   (BC)
  p3: '#f97316',   // 橙   — Offline-RL (CQL)
  p4: '#a78bfa',   // 紫   — Model-RL   (MBRL)
};

// ─── 迷你折线图 ───────────────────────────────────────────────────────────────
function Sparkline({
  data, w = 200, h = 50, color = C.accent, label, targetLine,
}: {
  data: number[]; w?: number; h?: number; color?: string;
  label?: string; targetLine?: number;
}) {
  if (!data || data.length < 2) {
    return (
      <div style={{ width: w, height: h, display: 'flex', alignItems: 'center',
        justifyContent: 'center', color: C.muted, fontSize: 10 }}>
        暂无数据
      </div>
    );
  }
  const min = Math.min(...data), max = Math.max(...data), rng = max - min || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / rng) * (h - 8) - 4;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const last = data[data.length - 1];
  const lp = pts.split(' ').pop()!.split(',');
  let targetY: number | null = null;
  if (targetLine !== undefined) {
    targetY = h - ((targetLine - min) / rng) * (h - 8) - 4;
    if (targetY < 0 || targetY > h) targetY = null;
  }
  return (
    <div style={{ position: 'relative', width: w, height: h }}>
      <svg width={w} height={h} style={{ display: 'block' }}>
        {targetY !== null && (
          <line x1={0} y1={targetY} x2={w} y2={targetY}
            stroke={C.success} strokeWidth={1} strokeDasharray="3,3" opacity={0.6} />
        )}
        <polyline points={pts} fill="none" stroke={color}
          strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
        <circle cx={parseFloat(lp[0])} cy={parseFloat(lp[1])} r={2.5} fill={color} />
      </svg>
      {label && (
        <div style={{ position: 'absolute', bottom: 0, right: 0, fontSize: 9, color: C.muted }}>
          {label}: <span style={{ color }}>{last.toFixed(2)}</span>
        </div>
      )}
    </div>
  );
}

// ─── 轨迹图 ───────────────────────────────────────────────────────────────────
function TrajectoryPanel({ result, accentColor = C.accent }: {
  result: InferResult | null; accentColor?: string;
}) {
  if (!result) {
    return (
      <div style={{
        height: 140, display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: C.muted, fontSize: 12, background: C.panel2, borderRadius: 8,
        border: `1px dashed ${C.border}`,
      }}>
        点击推理按钮后显示轨迹
      </div>
    );
  }

  const traj = result.trajectory;
  const target = result.lawson_target ?? 1e27;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {/* 摘要行 */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {[
          { label: '总奖励',    val: result.total_reward >= 1000
              ? `${(result.total_reward/1000).toFixed(1)}K`
              : result.total_reward.toFixed(0), color: accentColor },
          { label: '步数',      val: String(result.n_steps), color: C.purple },
          { label: 'Lawson达成', val: result.lawson_achieved ? '✓ 是' : '✗ 否',
            color: result.lawson_achieved ? C.success : C.muted },
          { label: '是否破裂',  val: result.disrupted ? '⚠ 是' : '✓ 否',
            color: result.disrupted ? C.danger : C.success },
          { label: 'Lawson值',  val: result.final_lawson.toExponential(2),
            color: result.final_lawson >= target ? C.success : C.muted },
        ].map(item => (
          <div key={item.label} style={{
            background: C.panel, border: `1px solid ${C.border}`,
            borderRadius: 6, padding: '6px 10px', minWidth: 80,
          }}>
            <div style={{ fontSize: 9, color: C.muted, marginBottom: 2 }}>{item.label}</div>
            <div style={{ fontSize: 14, fontWeight: 700, color: item.color,
              fontVariantNumeric: 'tabular-nums' }}>
              {item.val}
            </div>
          </div>
        ))}
      </div>

      {/* Lawson 主图 */}
      <div>
        <div style={{ fontSize: 10, color: C.muted, marginBottom: 4 }}>
          劳森参数 n·T·τ
          <span style={{ color: C.success, marginLeft: 8 }}>—— 目标 {target.toExponential(1)}</span>
        </div>
        <Sparkline data={traj.lawson || []} w={500} h={70}
          color={accentColor} targetLine={target} label="Lawson" />
      </div>

      {/* 三小图 */}
      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
        <div>
          <div style={{ fontSize: 9, color: C.muted, marginBottom: 2 }}>q95</div>
          <Sparkline data={traj.q95 || []} w={150} h={48} color={accentColor} label="q95" />
        </div>
        <div>
          <div style={{ fontSize: 9, color: C.muted, marginBottom: 2 }}>β_N</div>
          <Sparkline data={traj.beta_N || []} w={150} h={48} color={C.warning} label="β_N" />
        </div>
        <div>
          <div style={{ fontSize: 9, color: C.muted, marginBottom: 2 }}>P_heat (MW)</div>
          <Sparkline data={(traj.P_heat || []).map(v => v/1e6)} w={150} h={48}
            color={C.purple} label="P(MW)" />
        </div>
      </div>
    </div>
  );
}

// ─── 辅助：运行推理（POST） ────────────────────────────────────────────────────
async function runInfer(
  endpoint: string,
  addLog: (m: string) => void,
  setLoading: (b: boolean) => void,
  setResult: (r: InferResult) => void,
  label: string,
) {
  setLoading(true);
  addLog(`[${label}] 推理中...`);
  try {
    const res = await fetch(endpoint, { method: 'POST' });
    if (!res.ok) { const e = await res.json(); addLog(`[${label}] 失败：${e.detail}`); return; }
    const data: InferResult = await res.json();
    setResult(data);
    addLog(`[${label}] 完成 | reward=${data.total_reward.toFixed(0)} | lawson=${data.lawson_achieved ? '✓' : '✗'} | steps=${data.n_steps}`);
  } catch (e: any) { addLog(`[${label}] 错误：${e.message}`); }
  finally { setLoading(false); }
}

// ─── Phase 1 Tab：PPO RL ──────────────────────────────────────────────────────
function Phase1Tab({ phase, addLog }: {
  phase: PhaseStatus['phase1'] | null; addLog: (m: string) => void;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const [result,   setResult]   = useState<InferResult | null>(null);
  const [loading,  setLoading]  = useState(false);

  useEffect(() => {
    if (phase?.best_model && !selected) setSelected(phase.best_model);
  }, [phase]);

  const fmtRew = (v: number | null) =>
    v === null ? '—' : v >= 1000 ? `${(v/1000).toFixed(1)}K` : v.toFixed(0);

  const run = async () => {
    if (!selected) return;
    const url = `${BASE_URL}/api/rl/infer/best?model_path=${encodeURIComponent(selected)}`;
    await runInfer(url, addLog, setLoading, setResult, 'Phase1 PPO');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* 阶段说明 */}
      <div style={{
        padding: '10px 14px', background: '#071428',
        border: `1px solid #1a3a5c`, borderRadius: 8,
        borderLeft: `3px solid ${C.p1}`,
      }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: C.p1, marginBottom: 4 }}>
          Phase 1 · Sim-RL — 仿真环境在线强化学习（PPO）
        </div>
        <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.7 }}>
          完全不依赖真实数据，用 stable-baselines3 PPO 在 FusionEnv ODE 仿真环境中从零训练。
          Lawson 准则（n·T·τ &gt; 1e27）驱动奖励，4 个破裂条件终止 episode。
          当前已训练 <span style={{ color: C.p1 }}>{phase?.n_versions ?? 0} 个版本</span>。
        </div>
      </div>

      {/* 模型版本表 */}
      <div>
        <div style={{ fontSize: 11, fontWeight: 600, color: C.text, marginBottom: 8 }}>
          已训练模型版本（按最佳奖励降序）
        </div>
        <div style={{
          background: C.panel2, border: `1px solid ${C.border}`,
          borderRadius: 8, overflow: 'hidden',
        }}>
          <div style={{
            display: 'grid', gridTemplateColumns: '70px 90px 80px 70px 60px 60px',
            fontSize: 10, color: C.muted,
            padding: '7px 14px', borderBottom: `1px solid ${C.border}`,
          }}>
            <div>版本</div><div>最佳奖励</div><div>末轮奖励</div>
            <div>劳森达成</div><div>记录数</div><div>操作</div>
          </div>
          {!phase?.models?.length ? (
            <div style={{ padding: 14, fontSize: 11, color: C.dim }}>暂无模型</div>
          ) : phase.models.map(m => {
            const path = m.final_path || m.latest_ckpt;
            const isSel = selected === path;
            return (
              <div key={m.version} onClick={() => path && setSelected(path)}
                style={{
                  display: 'grid', gridTemplateColumns: '70px 90px 80px 70px 60px 60px',
                  fontSize: 11, color: C.text, padding: '8px 14px',
                  borderBottom: `1px solid ${C.border}`,
                  background: isSel ? '#0b1e38' : 'transparent', cursor: 'pointer',
                  transition: 'background 0.15s',
                }}>
                <div style={{ color: isSel ? C.p1 : C.text, fontWeight: isSel ? 700 : 400 }}>
                  {m.version}
                </div>
                <div style={{ color: C.success, fontVariantNumeric: 'tabular-nums' }}>
                  {fmtRew(m.best_reward)}
                </div>
                <div style={{ color: C.muted, fontVariantNumeric: 'tabular-nums' }}>
                  {fmtRew(m.last_reward)}
                </div>
                <div style={{ color: m.lawson_rate ? C.success : C.dim }}>
                  {m.lawson_rate !== null ? `${(m.lawson_rate*100).toFixed(0)}%` : '—'}
                </div>
                <div style={{ color: C.dim }}>{m.n_records}</div>
                <div style={{ fontSize: 10, color: isSel ? C.p1 : C.dim }}>
                  {isSel ? '● 已选' : '选择'}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 推理按钮 */}
      <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
        <div style={{ fontSize: 10, color: C.muted, flex: 1 }}>
          {selected ? `推理：${selected.split('/').slice(-3).join('/')}` : '请选择模型'}
        </div>
        <button onClick={run} disabled={loading || !selected} style={{
          padding: '8px 20px', borderRadius: 6, border: 'none', cursor: 'pointer',
          background: loading ? '#1e293b' : C.p1,
          color: loading ? C.muted : '#020817',
          fontWeight: 700, fontSize: 12, opacity: !selected ? 0.4 : 1,
        }}>
          {loading ? '推理中...' : '▶ 运行推理'}
        </button>
      </div>

      <TrajectoryPanel result={result} accentColor={C.p1} />
    </div>
  );
}

// ─── Phase 2 Tab：SFT + PPO ───────────────────────────────────────────────────
function Phase2Tab({ phase, phase1, addLog }: {
  phase: PhaseStatus['phase2'] | null;
  phase1: PhaseStatus['phase1'] | null;
  addLog: (m: string) => void;
}) {
  const [result,  setResult]  = useState<InferResult | null>(null);
  const [loading, setLoading] = useState(false);

  const demoModel = phase?.demo_model || phase?.ppo_warmstart_path || phase1?.best_model;

  const run = async () => {
    if (!demoModel) { addLog('[Phase2] 无可用模型'); return; }
    const url = `${BASE_URL}/api/rl/infer/best?model_path=${encodeURIComponent(demoModel)}`;
    await runInfer(url, addLog, setLoading, setResult, 'Phase2 SFT+PPO');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* 阶段说明 */}
      <div style={{
        padding: '10px 14px', background: '#161005',
        border: `1px solid #3d2800`, borderRadius: 8,
        borderLeft: `3px solid ${C.p2}`,
      }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: C.p2, marginBottom: 4 }}>
          Phase 2 · Sim-SFT — 仿真环境监督微调（行为克隆 BC）
        </div>
        <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.7 }}>
          行为克隆（BC）：从 Phase 1 专家轨迹中监督学习动作，得到 SFT Actor。
          SFT Actor 可热启动 Phase 1 PPO，使 RL 从有意义的初始策略出发。
          <span style={{ color: C.warning }}> 质量门槛：BC Actor 在 FusionEnv 评估 mean_reward ≥ 20K，否则自动跳过。</span>
        </div>
      </div>

      {/* SFT 状态卡 */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        <div style={{
          background: C.panel, border: `1px solid ${phase?.sft_ready ? '#1a3d20' : C.border}`,
          borderRadius: 8, padding: '14px',
        }}>
          <div style={{ fontSize: 11, color: C.muted, marginBottom: 6 }}>SFT BC Actor 状态</div>
          <div style={{ fontSize: 28, fontWeight: 800, color: phase?.sft_ready ? C.success : C.dim }}>
            {phase?.sft_ready ? '✓' : '○'}
          </div>
          <div style={{ fontSize: 12, color: phase?.sft_ready ? C.success : C.muted, marginTop: 4 }}>
            {phase?.sft_ready ? `已训练 (${phase.sft_size_mb ?? '?'} MB)` : '未训练'}
          </div>
        </div>
        <div style={{
          background: C.panel, border: `1px solid ${C.border}`,
          borderRadius: 8, padding: '14px',
        }}>
          <div style={{ fontSize: 11, color: C.muted, marginBottom: 6 }}>训练流程</div>
          <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.8 }}>
            <div>• 数据：Phase 1 专家轨迹（PPO rollout）</div>
            <div>• 算法：BCActorMLP（3层 MLP + Tanh×0.1）</div>
            <div>• 输出：p2_sft_actor.pt → Phase 1 PPO 热启动</div>
            <div style={{ color: C.warning }}>• 门槛：评估 ≥ 20K 才启用，否则跳过</div>
          </div>
        </div>
      </div>

      {/* Phase 2 PPO 模型（BC 热启动版本列表） */}
      {phase?.ppo_warmstart_ready ? (
        <div>
          <div style={{ fontSize: 11, fontWeight: 600, color: C.text, marginBottom: 8 }}>
            BC 热启动 PPO 模型
          </div>
          <div style={{
            background: C.panel2, border: `1px solid ${C.border}`,
            borderRadius: 8, overflow: 'hidden',
          }}>
            <div style={{
              display: 'grid', gridTemplateColumns: '70px 90px 80px 70px 60px',
              fontSize: 10, color: C.muted, padding: '7px 14px',
              borderBottom: `1px solid ${C.border}`,
            }}>
              <div>版本</div><div>最佳奖励</div><div>末轮奖励</div>
              <div>劳森达成</div><div>标签</div>
            </div>
            {phase.ppo_models?.map(m => (
              <div key={m.version} style={{
                display: 'grid', gridTemplateColumns: '70px 90px 80px 70px 60px',
                fontSize: 11, color: C.text, padding: '8px 14px',
                borderBottom: `1px solid ${C.border}`,
                background: demoModel === (m.final_path || m.latest_ckpt) ? '#1a1000' : 'transparent',
              }}>
                <div style={{ color: C.p2, fontWeight: 700 }}>{m.version}</div>
                <div style={{ color: C.success, fontVariantNumeric: 'tabular-nums' }}>
                  {m.best_reward !== null
                    ? m.best_reward >= 1000 ? `${(m.best_reward/1000).toFixed(1)}K` : m.best_reward.toFixed(0)
                    : '—'}
                </div>
                <div style={{ color: C.muted, fontVariantNumeric: 'tabular-nums' }}>
                  {m.last_reward !== null
                    ? m.last_reward >= 1000 ? `${(m.last_reward/1000).toFixed(1)}K` : m.last_reward.toFixed(0)
                    : '—'}
                </div>
                <div style={{ color: m.lawson_rate ? C.success : C.dim }}>
                  {m.lawson_rate !== null ? `${(m.lawson_rate*100).toFixed(0)}%` : '—'}
                </div>
                <div style={{
                  fontSize: 9, padding: '1px 5px', borderRadius: 3,
                  background: '#261800', color: C.p2, display: 'inline-block',
                }}>SFT→PPO</div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div style={{
          padding: '12px', background: C.panel,
          border: `1px solid ${C.border}`, borderRadius: 8,
          fontSize: 10, color: C.dim,
        }}>
          暂无 Phase 2 PPO 模型（请先完成 SFT 预训练，再使用 --ppo_warmstart 热启动 PPO 训练）
        </div>
      )}

      {/* 推理按钮 */}
      <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
        <div style={{ fontSize: 10, color: C.muted, flex: 1 }}>
          {demoModel
            ? `推理：${demoModel.split('/').slice(-3).join('/')}`
            : '暂无 Phase 2 模型'}
          {phase?.ppo_warmstart_ready && (
            <span style={{
              fontSize: 9, marginLeft: 8, padding: '1px 6px', borderRadius: 3,
              background: '#261800', color: C.p2,
            }}>SFT 热启动</span>
          )}
        </div>
        <button onClick={run} disabled={loading || !demoModel} style={{
          padding: '8px 20px', borderRadius: 6, border: 'none', cursor: 'pointer',
          background: loading ? '#1e293b' : C.p2,
          color: loading ? C.muted : '#020817',
          fontWeight: 700, fontSize: 12, opacity: !demoModel ? 0.4 : 1,
        }}>
          {loading ? '推理中...' : '▶ 运行推理'}
        </button>
      </div>

      <TrajectoryPanel result={result} accentColor={C.p2} />
    </div>
  );
}

// ─── Phase 3 Tab：Offline CQL ─────────────────────────────────────────────────
function Phase3Tab({ phase, addLog }: {
  phase: PhaseStatus['phase3'] | null; addLog: (m: string) => void;
}) {
  const [result,  setResult]  = useState<InferResult | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    await runInfer(`${BASE_URL}/api/rl/infer/offline`, addLog, setLoading, setResult, 'Phase3 CQL');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* 阶段说明 */}
      <div style={{
        padding: '10px 14px', background: '#150a00',
        border: `1px solid #3d1500`, borderRadius: 8,
        borderLeft: `3px solid ${C.p3}`,
      }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: C.p3, marginBottom: 4 }}>
          Phase 3 · Offline-RL — 历史数据离线强化学习（CQL）
        </div>
        <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.7 }}>
          Conservative Q-Learning（CQL）离线强化学习。
          数据来源：<span style={{ color: C.p3 }}>Phase 1 专家轨迹</span> ＋
          <span style={{ color: C.p4 }}> Phase 4b 回流轨迹</span>（数据飞轮核心）。
          device='cpu' 防止 Apple Silicon MPS silent crash。奖励 z-score 归一化防 Q 值爆炸。
        </div>
      </div>

      {/* 状态卡 */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        <div style={{
          background: C.panel, border: `1px solid ${phase?.ready ? '#2d1500' : C.border}`,
          borderRadius: 8, padding: '14px',
        }}>
          <div style={{ fontSize: 11, color: C.muted, marginBottom: 6 }}>CQL 模型状态</div>
          <div style={{ fontSize: 28, fontWeight: 800, color: phase?.ready ? C.success : C.dim }}>
            {phase?.ready ? '✓' : '○'}
          </div>
          <div style={{ fontSize: 12, color: phase?.ready ? C.success : C.muted, marginTop: 4 }}>
            {phase?.ready ? `已训练：${phase.cql_path?.split('/').pop()}` : '暂未训练'}
          </div>
        </div>

        <div style={{
          background: C.panel, border: `1px solid ${C.border}`,
          borderRadius: 8, padding: '14px',
        }}>
          <div style={{ fontSize: 11, color: C.muted, marginBottom: 6 }}>数据飞轮输入</div>
          <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.8 }}>
            <div style={{ color: C.p1 }}>→ Phase 1 专家轨迹（PPO rollout）</div>
            <div style={{ color: C.p4 }}>→ Phase 4b 回流轨迹（buffer.npz）</div>
            <div>• 200K steps，device='cpu'</div>
            <div>• z-score 归一化，conservative_weight=1.0</div>
          </div>
        </div>
      </div>

      {/* 飞轮说明 */}
      <div style={{
        padding: '12px', background: '#0d0a18',
        border: `1px solid #2a1a3a`, borderRadius: 8,
        fontSize: 10, color: C.dim, lineHeight: 1.8,
      }}>
        <div style={{ fontWeight: 600, color: C.muted, marginBottom: 4 }}>
          【数据飞轮机制】
        </div>
        <div>Phase 1 PPO → <span style={{ color: C.p3 }}>Phase 3 CQL</span> → Phase 4a Dyna → Phase 4b 精调</div>
        <div style={{ color: C.p4, marginTop: 4 }}>Phase 4b 产出的高质量轨迹（195K+ 水平）↩ 回流至 Phase 3</div>
        <div style={{ color: C.muted, marginTop: 4 }}>每轮迭代：CQL 学到更好策略 → MBRL 热启更强 → 精调更高 → 更好轨迹回流</div>
      </div>

      {/* 推理按钮 */}
      <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
        <div style={{ fontSize: 10, color: C.muted, flex: 1 }}>
          {phase?.ready
            ? `推理：${phase.cql_path?.split('/').slice(-2).join('/')}`
            : '暂无 CQL 模型（离线训练完成后自动解锁）'}
        </div>
        <button onClick={run} disabled={loading || !phase?.ready} style={{
          padding: '8px 20px', borderRadius: 6, border: 'none', cursor: 'pointer',
          background: loading ? '#1e293b' : C.p3,
          color: loading ? C.muted : '#020817',
          fontWeight: 700, fontSize: 12, opacity: !phase?.ready ? 0.4 : 1,
        }}>
          {loading ? '推理中...' : '▶ 运行推理'}
        </button>
      </div>

      <TrajectoryPanel result={result} accentColor={C.p3} />
    </div>
  );
}

// ─── Phase 4 Tab：World Model + Dyna MBRL ─────────────────────────────────────
function Phase4Tab({ phase, addLog }: {
  phase: PhaseStatus['phase4'] | null; addLog: (m: string) => void;
}) {
  const [result,  setResult]  = useState<InferResult | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    if (!phase?.mbrl_path) { addLog('[Phase4] 暂无 MBRL 模型'); return; }
    const url = `${BASE_URL}/api/rl/infer/best?model_path=${encodeURIComponent(phase.mbrl_path)}`;
    await runInfer(url, addLog, setLoading, setResult, 'Phase4 MBRL');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* 阶段说明 */}
      <div style={{
        padding: '10px 14px', background: '#0e0818',
        border: `1px solid #2d1a4a`, borderRadius: 8,
        borderLeft: `3px solid ${C.p4}`,
      }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: C.p4, marginBottom: 4 }}>
          Phase 4 · Model-RL — 世界模型强化学习（Dyna MBRL）
        </div>
        <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.7 }}>
          <strong style={{ color: C.p4 }}>4a Dyna：</strong>DeepONet MLP Ensemble 世界模型内 100 轮 Dyna 迭代，加速探索。
          <strong style={{ color: C.p4 }}> 4b 精调：</strong>FusionEnv 真实 ODE 环境直接精调，消除 World Model Bias。冠军 195.8K，Lawson 达成率 100%。
          精调产出轨迹回流 Phase 3，构成数据飞轮。
        </div>
      </div>

      {/* 冠军进度卡 */}
      {(() => {
        const champion = phase?.best_reward ?? 0;
        const target = 200000;
        const pct = Math.min(champion / target * 100, 100);
        return (
          <div style={{
            padding: '14px 16px', background: '#0a0718',
            border: `1px solid ${champion >= target ? C.success : '#3d2d6a'}`,
            borderRadius: 8,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
              <div>
                <div style={{ fontSize: 10, color: C.muted, marginBottom: 2 }}>当前冠军 mean_reward</div>
                <div style={{ fontSize: 28, fontWeight: 800, color: champion >= target ? C.success : C.p4,
                  fontVariantNumeric: 'tabular-nums' }}>
                  {champion >= 1000 ? `${(champion/1000).toFixed(1)}K` : (champion || '—')}
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: 10, color: C.muted, marginBottom: 2 }}>目标</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: C.muted }}>200K</div>
                <div style={{ fontSize: 10, color: champion >= target ? C.success : C.warning, marginTop: 2 }}>
                  {champion >= target ? '🏆 已达成！' : `差 ${(target - champion).toFixed(0)}`}
                </div>
              </div>
            </div>
            <div style={{ background: '#1e1535', borderRadius: 4, height: 8, overflow: 'hidden' }}>
              <div style={{
                width: `${pct}%`, height: '100%', borderRadius: 4,
                background: champion >= target
                  ? `linear-gradient(90deg, ${C.success}, #34d399)`
                  : `linear-gradient(90deg, ${C.p4}, #c084fc)`,
                transition: 'width 0.5s ease',
              }} />
            </div>
            <div style={{ fontSize: 9, color: C.dim, marginTop: 4 }}>{pct.toFixed(1)}% of target</div>
          </div>
        );
      })()}

      {/* 双状态卡 */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        {/* World Model 状态 */}
        <div style={{
          background: C.panel,
          border: `1px solid ${phase?.world_model_ready ? '#2d1a4a' : C.border}`,
          borderRadius: 8, padding: '14px',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div style={{ fontSize: 11, color: C.muted }}>World Model</div>
            <div style={{
              fontSize: 10, padding: '2px 6px', borderRadius: 4,
              background: phase?.world_model_ready ? '#1a0d2e' : '#1e293b',
              color: phase?.world_model_ready ? C.p4 : C.dim,
            }}>
              {phase?.world_model_ready ? '✓ 已训练' : '○ 未训练'}
            </div>
          </div>
          <div style={{ fontSize: 28, fontWeight: 800, color: phase?.world_model_ready ? C.p4 : C.dim, margin: '8px 0' }}>
            MLP ×5
          </div>
          <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.7 }}>
            {phase?.world_model_ready
              ? `${phase.world_model_size_mb} MB · 5 个独立 MLP · 输出 (mean, var)`
              : '5 独立 MLP · 输入 (state[7]+action[3]) → 输出 (next_state[7]+reward[1], var)'}
          </div>
        </div>

        {/* Dyna MBRL 状态 */}
        <div style={{
          background: C.panel,
          border: `1px solid ${phase?.mbrl_ready ? '#2d1a4a' : C.border}`,
          borderRadius: 8, padding: '14px',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div style={{ fontSize: 11, color: C.muted }}>
              Dyna MBRL {phase?.version ? <span style={{ color: C.p4 }}>{phase.version}</span> : ''}
            </div>
            <div style={{
              fontSize: 10, padding: '2px 6px', borderRadius: 4,
              background: phase?.mbrl_ready ? '#1a0d2e' : '#1e293b',
              color: phase?.mbrl_ready ? C.p4 : C.dim,
            }}>
              {phase?.mbrl_ready ? '✓ 已训练' : '○ 未训练'}
            </div>
          </div>
          {phase?.best_reward != null ? (
            <>
              <div style={{ fontSize: 28, fontWeight: 800, color: C.p4, margin: '8px 0 2px' }}>
                {(phase.best_reward / 1000).toFixed(1)}K
              </div>
              <div style={{ fontSize: 9, color: C.muted, marginBottom: 6 }}>mean reward / 100 ep</div>
              <div style={{ display: 'flex', gap: 8 }}>
                <div style={{
                  fontSize: 10, padding: '2px 8px', borderRadius: 4,
                  background: '#0d1a0d', color: C.success, fontVariantNumeric: 'tabular-nums',
                }}>
                  ⚡ Lawson {((phase.best_lawson ?? 0) * 100).toFixed(0)}%
                </div>
                <div style={{
                  fontSize: 10, padding: '2px 8px', borderRadius: 4,
                  background: '#0d1425', color: C.dim, fontVariantNumeric: 'tabular-nums',
                }}>
                  破裂 {((phase.disrupt_rate ?? 0) * 100).toFixed(0)}%
                </div>
              </div>
            </>
          ) : (
            <>
              <div style={{ fontSize: 28, fontWeight: 800, color: phase?.mbrl_ready ? C.p4 : C.dim, margin: '8px 0' }}>
                Dyna
              </div>
              <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.7 }}>
                World Model 采样虚拟轨迹 → PPO 策略更新 → 真实评估反馈
              </div>
            </>
          )}
        </div>
      </div>

      {/* Dyna 架构示意 */}
      <div style={{
        padding: '12px 14px', background: '#0a0d18',
        border: `1px solid ${C.border}`, borderRadius: 8,
      }}>
        <div style={{ fontSize: 10, color: C.muted, marginBottom: 8, fontWeight: 600 }}>
          Dyna 训练循环
        </div>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          fontSize: 10, color: C.dim, flexWrap: 'wrap',
        }}>
          {[
            { label: 'EAST 数据', color: C.p3 },
            { arrow: '→' },
            { label: 'World Model 更新', color: C.p4 },
            { arrow: '→' },
            { label: '虚拟轨迹采样 ×10K', color: C.p4 },
            { arrow: '→' },
            { label: 'PPO 策略更新', color: C.p1 },
            { arrow: '→' },
            { label: 'FusionEnv 真实评估', color: C.success },
            { arrow: '↺ 每 1000 步' },
          ].map((item, i) => (
            'arrow' in item
              ? <span key={i} style={{ color: C.muted }}>{item.arrow}</span>
              : <span key={i} style={{
                  padding: '3px 8px', borderRadius: 4,
                  background: '#0d1425',
                  border: `1px solid ${item.color}40`,
                  color: item.color,
                }}>
                  {item.label}
                </span>
          ))}
        </div>
      </div>

      {/* 推理按钮 */}
      <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
        <div style={{ fontSize: 10, color: C.muted, flex: 1 }}>
          {phase?.mbrl_ready
            ? `推理：${phase.mbrl_path?.split('/').slice(-2).join('/')}`
            : 'Dyna MBRL 模型训练完成后自动解锁'}
        </div>
        <button onClick={run} disabled={loading || !phase?.mbrl_ready} style={{
          padding: '8px 20px', borderRadius: 6, border: 'none', cursor: 'pointer',
          background: loading ? '#1e293b' : C.p4,
          color: loading ? C.muted : '#020817',
          fontWeight: 700, fontSize: 12, opacity: !phase?.mbrl_ready ? 0.4 : 1,
        }}>
          {loading ? '推理中...' : '▶ 运行推理'}
        </button>
      </div>

      <TrajectoryPanel result={result} accentColor={C.p4} />
    </div>
  );
}

// ─── 主组件 ───────────────────────────────────────────────────────────────────
type PhaseTab = 'p1' | 'p2' | 'p3' | 'p4';

export function RLDashboard() {
  const [rlStatus,     setRlStatus]     = useState<RLStatus | null>(null);
  const [history,      setHistory]      = useState<HistoryRecord[]>([]);
  const [phaseStatus,  setPhaseStatus]  = useState<PhaseStatus | null>(null);
  const [log,          setLog]          = useState<string[]>([]);
  const [activePhase,  setActivePhase]  = useState<PhaseTab>('p1');

  const wsRef   = useRef<WebSocket | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const logRef  = useRef<HTMLDivElement | null>(null);

  const addLog = useCallback((msg: string) => {
    setLog(prev => [...prev.slice(-99), `[${new Date().toLocaleTimeString()}] ${msg}`]);
  }, []);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  const fetchStatus = useCallback(async () => {
    try {
      const r = await fetch(`${BASE_URL}/api/rl/status`);
      if (r.ok) setRlStatus(await r.json());
    } catch {}
  }, []);

  const fetchHistory = useCallback(async () => {
    try {
      const r = await fetch(`${BASE_URL}/api/rl/history`);
      if (r.ok) { const d = await r.json(); setHistory(d.history || []); }
    } catch {}
  }, []);

  const fetchPhaseStatus = useCallback(async () => {
    try {
      const r = await fetch(`${BASE_URL}/api/rl/phase-status`);
      if (r.ok) setPhaseStatus(await r.json());
    } catch {}
  }, []);

  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    const _p = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const _pfx = window.location.pathname.startsWith('/jpfusion') ? '/jpfusion' : '';
    const ws = new WebSocket(`${_p}//${window.location.host}${_pfx}/api/rl/ws`);
    ws.onopen    = () => addLog('WebSocket 已连接');
    ws.onclose   = () => {};
    ws.onerror   = () => addLog('WebSocket 连接失败');
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === 'rl_init') setHistory(msg.history || []);
        else if (msg.type === 'rl_progress') {
          setHistory(prev => [...prev, msg]);
          addLog(`step=${msg.timestep} | rew=${msg.mean_reward?.toFixed(0)} | disrupt=${(msg.disruption_rate*100).toFixed(0)}% | lawson=${(msg.lawson_rate*100).toFixed(0)}%`);
        }
      } catch {}
    };
    wsRef.current = ws;
  }, [addLog]);

  useEffect(() => {
    fetchStatus(); fetchHistory(); fetchPhaseStatus(); connectWS();
    pollRef.current = setInterval(() => {
      fetchStatus(); fetchHistory(); fetchPhaseStatus();
    }, 8000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      wsRef.current?.close();
    };
  }, [fetchStatus, fetchHistory, fetchPhaseStatus, connectWS]);

  const isTraining   = rlStatus?.status === 'training';
  const latestHist   = history[history.length - 1];
  const sColor       = { idle: C.muted, training: C.warning, done: C.success, error: C.danger };
  const statusColor  = sColor[rlStatus?.status ?? 'idle'] ?? C.muted;

  // Phase Tab 配置
  const phaseTabs: { key: PhaseTab; label: string; color: string; icon: string; ready: boolean }[] = [
    { key: 'p1', label: 'Phase 1 · Sim-RL',      color: C.p1, icon: '🔵', ready: phaseStatus?.phase1.ready ?? false },
    { key: 'p2', label: 'Phase 2 · Sim-SFT',     color: C.p2, icon: '🟡', ready: phaseStatus?.phase2.sft_ready ?? false },
    { key: 'p3', label: 'Phase 3 · Offline-RL',  color: C.p3, icon: '🟠', ready: phaseStatus?.phase3.ready ?? false },
    { key: 'p4', label: 'Phase 4 · Model-RL',    color: C.p4, icon: '🟣', ready: phaseStatus?.phase4.world_model_ready ?? false },
  ];

  return (
    <div style={{
      display: 'flex', height: '100%', overflow: 'hidden',
      background: C.bg, color: C.text,
      fontFamily: "'Inter', 'SF Pro Display', -apple-system, sans-serif",
    }}>

      {/* ══════════════════════════════════════════════════════
          左栏：训练监控（340px）
         ══════════════════════════════════════════════════════ */}
      <div style={{
        width: 320, flexShrink: 0,
        borderRight: `1px solid ${C.border}`,
        overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 12,
        padding: '14px 12px',
      }}>

        {/* 标题 + 状态 */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, color: C.text }}>FusionRL 控制台</div>
            <div style={{ fontSize: 10, color: C.muted }}>托卡马克等离子体 RL 控制</div>
          </div>
          <div style={{
            fontSize: 10, padding: '3px 8px', borderRadius: 4,
            background: '#1e293b', color: statusColor,
          }}>
            {rlStatus?.status ?? 'idle'}
            {isTraining && ` · ${(rlStatus?.total_steps ?? 0).toLocaleString()} 步`}
          </div>
        </div>

        {/* 四阶段总览条 */}
        <div style={{
          background: C.panel, border: `1px solid ${C.border}`,
          borderRadius: 8, padding: '10px 12px',
        }}>
          <div style={{ fontSize: 10, color: C.muted, marginBottom: 8 }}>四阶段训练进度</div>
          <div style={{ display: 'flex', gap: 6 }}>
            {phaseTabs.map(pt => (
              <div key={pt.key} style={{
                flex: 1, padding: '6px 4px', borderRadius: 6, textAlign: 'center',
                background: pt.ready ? `${pt.color}18` : '#1e293b',
                border: `1px solid ${pt.ready ? pt.color + '40' : C.border}`,
                cursor: 'pointer',
              }} onClick={() => setActivePhase(pt.key)}>
                <div style={{ fontSize: 14 }}>{pt.icon}</div>
                <div style={{ fontSize: 8, color: pt.ready ? pt.color : C.dim, marginTop: 2 }}>
                  {pt.ready ? '✓' : '○'}
                </div>
              </div>
            ))}
          </div>
          <div style={{
            display: 'flex', gap: 4, marginTop: 6, flexWrap: 'wrap',
          }}>
            {phaseStatus && [
              { label: `P1 Sim-RL: ${phaseStatus.phase1.n_versions}版本`, color: C.p1 },
              { label: phaseStatus.phase2.sft_ready ? 'P2 Sim-SFT ✓' : 'P2: 待训练', color: C.p2 },
              { label: phaseStatus.phase3.ready ? 'P3 Offline-RL ✓' : 'P3: 待训练', color: C.p3 },
              { label: phaseStatus.phase4.world_model_ready ? 'P4 Model-RL ✓' : 'P4: 待训练', color: C.p4 },
            ].map((item, i) => (
              <span key={i} style={{
                fontSize: 9, padding: '1px 6px', borderRadius: 3,
                background: '#0d1425', color: item.color,
              }}>{item.label}</span>
            ))}
          </div>
        </div>

        {/* 实时训练指标 */}
        {latestHist && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {[
              { k: 'rew',     label: 'mean_rew', sub: '平均奖励',
                val: latestHist.mean_reward >= 1000
                  ? `${(latestHist.mean_reward/1000).toFixed(1)}K`
                  : latestHist.mean_reward.toFixed(0), color: C.accent },
              { k: 'len',     label: 'ep_len',   sub: '每局步数',
                val: latestHist.mean_ep_length.toFixed(0), color: C.purple },
              { k: 'dis',     label: 'disrupt',  sub: '破裂率',
                val: `${(latestHist.disruption_rate*100).toFixed(0)}%`, color: C.danger },
              { k: 'law',     label: 'lawson',   sub: '劳森达成',
                val: `${(latestHist.lawson_rate*100).toFixed(0)}%`, color: C.success },
              { k: 'gam',     label: 'gaming',   sub: 'Gaming 代理',
                val: latestHist.gaming_proxy.toFixed(3), color: C.warning },
            ].map(item => (
              <div key={item.k} style={{
                background: C.panel, border: `1px solid ${C.border}`,
                borderRadius: 6, padding: '7px 8px', flex: '1 1 80px', minWidth: 72,
              }}>
                <div style={{ fontSize: 9, color: C.muted }}>{item.label}</div>
                <div style={{ fontSize: 8, color: C.dim, marginBottom: 3 }}>{item.sub}</div>
                <div style={{ fontSize: 15, fontWeight: 700, color: item.color,
                  fontVariantNumeric: 'tabular-nums' }}>
                  {item.val}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* 奖励曲线 */}
        {history.length > 1 && (
          <div style={{
            background: C.panel, border: `1px solid ${C.border}`,
            borderRadius: 8, padding: 10,
          }}>
            <div style={{ fontSize: 10, fontWeight: 600, color: C.text, marginBottom: 6 }}>
              训练曲线
            </div>
            <Sparkline data={history.map(h => h.mean_reward)} w={270} h={55}
              color={C.accent} label="reward" />
            <div style={{ display: 'flex', gap: 10, marginTop: 6 }}>
              <Sparkline data={history.map(h => h.disruption_rate)} w={115} h={36}
                color={C.danger} label="disrupt" />
              <Sparkline data={history.map(h => h.lawson_rate)} w={115} h={36}
                color={C.success} label="lawson" />
            </div>
          </div>
        )}


        {/* 日志 */}
        <div style={{
          background: C.panel2, border: `1px solid ${C.border}`,
          borderRadius: 8, padding: 8, flex: 1, minHeight: 60,
        }}>
          <div style={{ fontSize: 9, color: C.muted, marginBottom: 3 }}>日志</div>
          <div ref={logRef} style={{ maxHeight: 180, overflowY: 'auto' }}>
            {log.length === 0
              ? <div style={{ fontSize: 9, color: C.dim }}>（暂无日志）</div>
              : log.map((l, i) => (
                <div key={i} style={{ fontSize: 8.5, color: C.muted, lineHeight: '1.7' }}>{l}</div>
              ))
            }
          </div>
        </div>

        {/* 名词备注 */}
        <div style={{
          background: '#0a1020', border: `1px solid #1e3a5f`,
          borderRadius: 8, padding: 10, fontSize: 8.5, color: '#475569', lineHeight: '1.8',
        }}>
          <div style={{ fontWeight: 600, color: '#64748b', marginBottom: 3 }}>【四阶段命名体系 · 名词备注】</div>
          <div style={{ color: '#38bdf8', fontWeight: 600, marginTop: 4 }}>Phase 1 · Sim-RL（仿真在线强化学习）</div>
          <div>PPO — 近端策略优化，在 FusionEnv 仿真器中实时交互学习；what：on-policy RL；why：无需历史数据，从零探索</div>
          <div>FusionEnv — 托卡马克等离子体 ODE 仿真环境；what：等离子体动力学模拟器；why：替代真实放电的安全沙盒</div>
          <div style={{ color: '#f59e0b', fontWeight: 600, marginTop: 4 }}>Phase 2 · Sim-SFT（仿真监督微调）</div>
          <div>SFT / BC — 监督微调 = 行为克隆；what：模仿专家动作的监督学习；why：给 PPO 有意义的初始策略，避免从随机策略探索</div>
          <div>热启动 — BC 权重 → PPO 初始化；what：迁移 SFT 学到的策略；why：加速 Phase 1 收敛 ~2×</div>
          <div style={{ color: '#f97316', fontWeight: 600, marginTop: 4 }}>Phase 3 · Offline-RL（历史数据离线强化学习）</div>
          <div>CQL — Conservative Q-Learning；what：离线 RL 算法；why：保守 Q 估计防止对历史数据外区域过度乐观</div>
          <div>Distribution Shift — 分布偏移；what：训练数据分布 ≠ 部署时状态分布；why：离线 RL 最核心挑战</div>
          <div style={{ color: '#a78bfa', fontWeight: 600, marginTop: 4 }}>Phase 4 · Model-RL（世界模型强化学习）</div>
          <div>MBRL / Dyna — 模型基强化学习；what：先建世界模型，再在想象中训练；why：减少真实仿真调用次数</div>
          <div>World Model — 世界模型；what：预测下一状态+奖励的 MLP Ensemble；why：提供虚拟经验扩展训练数据</div>
          <div style={{ color: '#10b981', fontWeight: 600, marginTop: 4 }}>通用术语</div>
          <div>Lawson Criterion — n·T·τ &gt; 1e27，核聚变点火门槛；达成 = episode 成功</div>
          <div>τ_E — 能量约束时间；what：等离子体保持热度的时间；why：Lawson 准则的核心参数之一</div>
          <div>破裂（Disruption）— 等离子体失控终止；what：episode 提前结束事件；why：真实托卡马克会损坏设备</div>
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════
          右栏：四阶段推理面板
         ══════════════════════════════════════════════════════ */}
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

        {/* Phase Tab 导航 */}
        <div style={{
          display: 'flex', gap: 0,
          borderBottom: `1px solid ${C.border}`,
          background: C.bg, flexShrink: 0,
        }}>
          {phaseTabs.map(pt => {
            const active = activePhase === pt.key;
            return (
              <button key={pt.key} onClick={() => setActivePhase(pt.key)} style={{
                padding: '14px 24px',
                border: 'none', borderBottom: active ? `2px solid ${pt.color}` : '2px solid transparent',
                background: active ? `${pt.color}0e` : 'transparent',
                color: active ? pt.color : C.muted,
                cursor: 'pointer', fontSize: 12, fontWeight: active ? 700 : 400,
                display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3,
                transition: 'all 0.15s',
              }}>
                <div style={{ fontSize: 18 }}>{pt.icon}</div>
                <div>{pt.label}</div>
                <div style={{
                  fontSize: 9, color: pt.ready ? pt.color : C.dim,
                }}>
                  {pt.ready ? '● 已就绪' : '○ 未训练'}
                </div>
              </button>
            );
          })}
        </div>

        {/* Tab 内容 */}
        <div style={{ flex: 1, overflowY: 'auto', padding: 20 }}>
          {activePhase === 'p1' && (
            <Phase1Tab phase={phaseStatus?.phase1 ?? null} addLog={addLog} />
          )}
          {activePhase === 'p2' && (
            <Phase2Tab
              phase={phaseStatus?.phase2 ?? null}
              phase1={phaseStatus?.phase1 ?? null}
              addLog={addLog}
            />
          )}
          {activePhase === 'p3' && (
            <Phase3Tab phase={phaseStatus?.phase3 ?? null} addLog={addLog} />
          )}
          {activePhase === 'p4' && (
            <Phase4Tab phase={phaseStatus?.phase4 ?? null} addLog={addLog} />
          )}
        </div>
      </div>
    </div>
  );
}

export default RLDashboard;
