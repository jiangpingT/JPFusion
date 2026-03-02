/**
 * client.ts — 后端 API 调用封装
 */

import axios from 'axios';

const BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000,
});

// ─── 类型定义 ───────────────────────────────────────────────────────────────

export interface GenerateDataParams {
  n_samples:      number;
  grid_size:      number;
  add_turbulence: boolean;
}

export interface TrainParams {
  n_epochs:    number;
  batch_size:  number;
  lr:          number;
  max_samples: number;
  // PINN 物理约束参数
  use_pinn:    boolean;
  lambda_pde:  number;
  lambda_mono: number;
  lambda_sym:  number;
  lambda_bc:   number;
}

export interface InferenceParams {
  n_e:       number;
  T_e:       number;
  B:         number;
  grid_size: number;
}

export interface GsMeta {
  p_axis:         number;
  I_p_input:      number;
  plasma_current: number;
  poloidal_beta:  number;
  psi_axis:       number;
  psi_bndry:      number;
  q_axis:         number | null;
  q95:            number | null;
  R0:             number;
  a:              number;
}

export interface InferenceResult {
  source:             'model' | 'physics' | 'physics_gs';
  x:                  number[][];
  y:                  number[][];
  T_values:           number[][];
  T_uncertainty:      number[][] | null;
  T_uncertainty_max:  number | null;
  T_uncertainty_mean: number | null;
  psiN_profile:       number[][] | null;
  gs_meta:            GsMeta | null;
  T_min:              number;
  T_max:              number;
  grid_size:          number;
  physics_params: {
    lambda_D: number;
    omega_p:  number;
    v_alfven: number;
    beta:     number;
  };
}

export interface ModelStatus {
  model_ready:      boolean;
  model_exists:     boolean;
  dataset_exists:   boolean;
  training_status:  string;
  current_epoch:    number;
  total_epochs:     number;
  train_loss:       number | null;
  val_loss:         number | null;
  best_val_loss:    number;
  error_msg:        string | null;
}

// ─── API 函数 ────────────────────────────────────────────────────────────────

export const fusionApi = {
  generateData: (params: GenerateDataParams) =>
    api.post('/api/generate-data', params).then(r => r.data),

  getGenerateStatus: () =>
    api.get('/api/generate-data/status').then(r => r.data),

  startTraining: (params: TrainParams) =>
    api.post('/api/train', params).then(r => r.data),

  inference: (params: InferenceParams): Promise<InferenceResult> =>
    api.post('/api/inference', params).then(r => r.data),

  getModelStatus: (): Promise<ModelStatus> =>
    api.get('/api/model-status').then(r => r.data),

  getPhysicsProfile: (params: InferenceParams): Promise<InferenceResult> =>
    api.post('/api/plasma-profile', params).then(r => r.data),

  getTrainingHistory: () =>
    api.get('/api/training-history').then(r => r.data),

  getGsProfile: (params: InferenceParams): Promise<InferenceResult> =>
    api.post('/api/plasma-profile-gs', params).then(r => r.data),
};

export const WS_URL = 'ws://localhost:8000/ws/training-progress';
