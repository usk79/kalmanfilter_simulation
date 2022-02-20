// カルマンフィルタによる状態推定器　まずは2次元限定

extern crate nalgebra as na;
use na::{U2, U3, Dynamic, ArrayStorage, VecStorage, Matrix, OMatrix, DMatrix};

#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    mat_a: DMatrix<f64>,
    mat_b: DMatrix<f64>,
    mat_c: DMatrix<f64>,
    x: DMatrix<f64>,    // 状態ベクトル
    mat_p: DMatrix<f64>,    // 誤差共分散行列
    mat_q: DMatrix<f64>,    // プロセスノイズの共分散行列
    mat_r: DMatrix<f64>,    // 観測ノイズの共分散行列
    state_dim: usize,   // 状態次数
    input_dim: usize,   // 入力次数
    output_dim: usize,  // 出力次数
}

impl Model {
    pub fn new(sdim: usize, idim: usize, odim: usize) -> Self {
        Model {
            mat_a: DMatrix::from_element(sdim, sdim, 0.0),
            mat_b: DMatrix::from_element(sdim, idim, 0.0),
            mat_c: DMatrix::from_element(odim, sdim, 0.0),
            x: DMatrix::from_element(sdim, 1, 0.0),
            mat_p: DMatrix::from_element(sdim, sdim, 0.0),
            mat_q: DMatrix::from_element(sdim, sdim, 0.0),
            mat_r: DMatrix::from_element(odim, odim, 0.0),
            state_dim: sdim,
            input_dim: idim,
            output_dim: odim,
        }
    }

    pub fn init_state(&mut self, init_state: Vec<f64>, init_p: Vec<f64>) -> Result<(), &str> {

        if init_state.len() != self.state_dim {
            return Err("状態変数のサイズが違います。");
        }

        if init_p.len() != self.state_dim * self.state_dim {
            return Err("誤差共分散行列の次元が違います。");
        }

        for (i, elem) in init_state.iter().enumerate() {
            self.x[i] = *elem;
        }

        for (i, elem) in init_p.iter().enumerate() {
            self.mat_p[i] = *elem;
        }
        
        Ok(())
    }

    pub fn set_mat_a(&mut self, mat_a: Vec<f64>) -> Result<(), &str> {

        if mat_a.len() != self.state_dim * self.state_dim {
            return Err("A行列のサイズが違います。");
        }

        for (i, elem) in mat_a.iter().enumerate() {
            self.mat_a[(i / self.state_dim, i % self.state_dim)] = *elem;
        }

        Ok(())
    }

    pub fn set_mat_b(&mut self, mat_b: Vec<f64>) -> Result<(), &str> {
        
        if mat_b.len() != self.state_dim * self.input_dim {
            return Err("B行列のサイズが違います。");
        }

        for (i, elem) in mat_b.iter().enumerate() {
            self.mat_b[(i / self.input_dim, i % self.input_dim)] = *elem;
        }

        Ok(())
    }

    pub fn set_mat_c(&mut self, mat_c: Vec<f64>) -> Result<(), &str> {

        if mat_c.len() != self.output_dim * self.state_dim {
            return Err("C行列のサイズが違います。");
        }

        for (i, elem) in mat_c.iter().enumerate() {
            self.mat_c[(i / self.state_dim, i % self.state_dim)] = *elem;
        }

        Ok(())
    }

    pub fn set_mat_q(&mut self, mat_q: Vec<f64>) -> Result<(), &str> {
        if mat_q.len() != self.state_dim * self.state_dim {
            return Err("プロセスノイズの共分散行列の次元が違います。");
        }

        for (i, elem) in mat_q.iter().enumerate() {
            self.mat_p[(i / self.state_dim, i % self.state_dim)] = *elem;
        }

        Ok(())
    }

    pub fn set_mat_r(&mut self, mat_r: Vec<f64>) -> Result<(), &str> {
        if mat_r.len() != self.output_dim * self.output_dim  {
            return Err("観測ノイズの共分散行列の次元が違います。");
        }

        for (i, elem) in mat_r.iter().enumerate() {
            self.mat_r[(i / self.output_dim, i % self.output_dim)] = *elem;
        }

        Ok(())
    }

    // 予測ステップ
    pub fn predict_nextstate(&mut self, input: Vec<f64>) -> Result<Vec<f64>, &str> {
        if input.len() != self.input_dim {
            return Err("入力の次数が違います。");
        }

        let inputmat = DMatrix::from_vec(self.input_dim, 1, input);
        
        self.x = &self.mat_a * &self.x + &self.mat_b * inputmat;
        self.mat_p = &self.mat_a * &self.mat_p * &self.mat_a.transpose() + &self.mat_q;

        let mut result = vec![0.0; self.state_dim];
        for (i, elem) in self.x.iter().enumerate() {
            result[i] = *elem;
        }

        Ok(result)
    }
    // 観測ステップ
    pub fn observation(&self, noise: Vec<f64>) -> Result<Vec<f64>, &str> {
        if noise.len() != self.output_dim {
            return Err("観測ノイズの次数が違います。");
        }
        let noisemat = DMatrix::from_vec(self.output_dim, 1, noise);

        let obs = &self.mat_c * &self.x + noisemat;

        let mut result = vec![0.0; self.output_dim];
        for (i, elem) in obs.iter().enumerate() {
            result[i] = *elem;
        }

        Ok(result)
    }

    // 更新ステップ
    pub fn update_nextstate(&mut self, observation: Vec<f64>) -> Result<Vec<f64>, &str> {
        if observation.len() != self.output_dim {
            return Err("観測ベクトルの次数が違います。");
        }
        let y = DMatrix::from_vec(self.output_dim, 1, observation);
        let y_dash =  &self.mat_c * &self.x;
        // カルマンゲインの計算

        let s = &self.mat_c * &self.mat_p * &self.mat_c.transpose() + &self.mat_r;
        
        let s_inv = s.try_inverse().unwrap();

        let k = &self.mat_p * &self.mat_c.transpose() * s_inv;

        // 状態ベクトルと誤差共分散行列の更新
        self.x = &self.x + &k * (y - y_dash);
        self.mat_p = &self.mat_p - &k * &self.mat_c * &self.mat_p;
 
        let mut result = vec![0.0; self.state_dim];
        for (i, elem) in self.x.iter().enumerate() {
            result[i] = *elem;
        }
        Ok(result)
    }
    // ノイズを考慮した状態遷移
    pub fn sim_nextstate(&mut self, input: Vec<f64>, noise: Vec<f64>) -> Result<Vec<f64>, &str> {
        if noise.len() != self.state_dim {
            return Err("プロセスノイズの次数が違います。");
        }
        let noisemat = DMatrix::from_vec(self.state_dim, 1, noise);

        if input.len() != self.input_dim {
            return Err("入力の次数が違います。");
        }

        let inputmat = DMatrix::from_vec(self.input_dim, 1, input);
        
        self.x = &self.mat_a * &self.x + &self.mat_b * inputmat + noisemat;

        let mut result = vec![0.0; self.state_dim];
        for (i, elem) in self.x.iter().enumerate() {
            result[i] = *elem;
        }

        Ok(result)
    }
}