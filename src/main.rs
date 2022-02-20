use rand::prelude::{Distribution, thread_rng};
use rand_distr::Normal;


mod plot;
use plot::{Points, scatter, TimeSeries, timeplot, output_timeseries};

mod modelestimator;
use modelestimator::Model;

fn main() {
    //simulation1();
    simulation2();
}

// 観測が位置と速度の場合
fn simulation1() {
    const SAMPLE_TIME: f64 = 0.01;
    const SIMULATION_TIME: f64 = 100.0;  // シミュレーション時間[s]
    const SIMULATION_STEPS: i32 = (SIMULATION_TIME / SAMPLE_TIME) as i32;
    
    const SIGMA2_POS: f64 = 0.01; // 入力ノイズの分散（位置）
    const SIGMA2_SPD: f64 = 0.01; // 入力ノイズの分散（速度）
    const SIGMA2_SPDSENS: f64 = 1.0; // 速度センサの観測ノイズの分散

    let mut model_true = create_model1(SAMPLE_TIME);  // 実際のモデル
    let mut model_est = create_model1(SAMPLE_TIME);  // 推定器
    let mut model_odo = create_model_odo(SAMPLE_TIME);  // デッドレコニング

    let mut rng = thread_rng();
    let inputdist_pos = Normal::<f64>::new(0.0, SIGMA2_POS).unwrap(); // 入力の確率密度関数
    let inputdist_spd = Normal::<f64>::new(0.0, SIGMA2_SPD).unwrap(); // 入力の確率密度関数
    let spdsensdist = Normal::<f64>::new(0.0, SIGMA2_SPDSENS).unwrap(); // 速度センサの確率密度関数

    // データ計測
    let mut x_true = TimeSeries::new(SAMPLE_TIME, SIMULATION_TIME, 0.0);
    let mut v_true = TimeSeries::new(SAMPLE_TIME, SIMULATION_TIME, 0.0);
    let mut x_odo = TimeSeries::new(SAMPLE_TIME, SIMULATION_TIME, 0.0);
    let mut v_odo = TimeSeries::new(SAMPLE_TIME, SIMULATION_TIME, 0.0);
    let mut x_est = TimeSeries::new(SAMPLE_TIME, SIMULATION_TIME, 0.0);
    let mut v_est = TimeSeries::new(SAMPLE_TIME, SIMULATION_TIME, 0.0);

    for step in 1..SIMULATION_STEPS {
      //  println!("Step {}\n", step);

        let time = step as f64 * SAMPLE_TIME;
        let accr = accr_generator(time); // 加速度の入力信号生成

        // 実モデルのシミュレーション
        let state_true = model_true.sim_nextstate(vec![accr + inputdist_pos.sample(&mut rng)], vec![inputdist_pos.sample(&mut rng), 0.0]).unwrap();
        x_true.recordvalue(state_true[0]).unwrap();
        v_true.recordvalue(state_true[1]).unwrap();

        // 観測
        let v_obs = model_true.observation(vec![spdsensdist.sample(&mut rng), spdsensdist.sample(&mut rng)]).unwrap();

        // 車輪速の積分によるデッドレコニング
        let state_odo = model_odo.predict_nextstate(vec![v_obs[1]]).unwrap();
        x_odo.recordvalue(state_odo[0]).unwrap();
        v_odo.recordvalue(state_odo[1]).unwrap();

        // カルマンフィルタによる推定
        let _state_predict = model_est.predict_nextstate(vec![accr]);
        let state_update = model_est.update_nextstate(v_obs).unwrap();
        x_est.recordvalue(state_update[0]).unwrap();
        v_est.recordvalue(state_update[1]).unwrap();
    }

    let last_step = SIMULATION_STEPS as usize - 1;
    println!("x_true = {}, x_est = {}, x_odo = {}", &x_true.getvalue_bystep(last_step).unwrap(), &x_est.getvalue_bystep(last_step).unwrap(), &x_odo.getvalue_bystep(last_step).unwrap());
    
    x_odo.output_csv("x_odo.csv");
    v_odo.output_csv("v_odo.csv");
    x_est.output_csv("x_est.csv");
    v_est.output_csv("v_est.csv");

    // 計測データのプロット
    let plots_x = vec![x_true, x_est, x_odo];
    let plots_v = vec![v_true, v_est, v_odo];
    timeplot(&plots_x, "plots_x.png", "plots_x");
    timeplot(&plots_v, "plots_v.png", "plots_v");


    
}

fn create_model1(sample_time: f64) -> Model {
    //let mut model = Model::new(2, 1, 1);
    let mut model = Model::new(2, 1, 2);

    model.init_state(vec![0.0, 0.0], vec![0.01, 0.0, 0.0, 0.001]).unwrap();
    model.set_mat_a(vec![1.0, sample_time, 0.0, 1.0]).unwrap();
    model.set_mat_b(vec![0.0, sample_time]).unwrap();
    
    model.set_mat_c(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    //model.set_mat_c(vec![0.0, 1.0]).unwrap();
    
    model.set_mat_q(vec![0.1, 0.0, 0.0, 0.1]).unwrap();
    
    model.set_mat_r(vec![0.05, 0.0, 0.0, 0.05]).unwrap();
    //model.set_mat_r(vec![0.1]).unwrap();

    model
}

// 観測が速度の場合
fn simulation2() {
    const SAMPLE_TIME: f64 = 0.01;
    const SIMULATION_TIME: f64 = 100.0;  // シミュレーション時間[s]
    const SIMULATION_STEPS: i32 = (SIMULATION_TIME / SAMPLE_TIME) as i32;
    
    const SIGMA2_POS: f64 = 0.01; // 入力ノイズの分散（位置）
    const SIGMA2_SPD: f64 = 0.01; // 入力ノイズの分散（速度）
    const SIGMA2_SPDSENS: f64 = 5.1; // 速度センサの観測ノイズの分散

    let mut model_true = create_model2(SAMPLE_TIME);  // 実際のモデル
    let mut model_est = create_model2(SAMPLE_TIME);  // 推定器
    let mut model_odo = create_model_odo(SAMPLE_TIME);  // デッドレコニング

    let mut rng = thread_rng();
    let inputdist_pos = Normal::<f64>::new(0.0, SIGMA2_POS).unwrap(); // 入力の確率密度関数
    let inputdist_spd = Normal::<f64>::new(0.0, SIGMA2_SPD).unwrap(); // 入力の確率密度関数
    let spdsensdist = Normal::<f64>::new(0.0, SIGMA2_SPDSENS).unwrap(); // 速度センサの確率密度関数

    // データ計測
    let mut x_true = TimeSeries::new(SAMPLE_TIME, SIMULATION_TIME, 0.0);
    let mut v_true = TimeSeries::new(SAMPLE_TIME, SIMULATION_TIME, 0.0);
    let mut x_odo = TimeSeries::new(SAMPLE_TIME, SIMULATION_TIME, 0.0);
    let mut v_odo = TimeSeries::new(SAMPLE_TIME, SIMULATION_TIME, 0.0);
    let mut x_est = TimeSeries::new(SAMPLE_TIME, SIMULATION_TIME, 0.0);
    let mut v_est = TimeSeries::new(SAMPLE_TIME, SIMULATION_TIME, 0.0);

    for step in 1..SIMULATION_STEPS {
      //  println!("Step {}\n", step);

        let time = step as f64 * SAMPLE_TIME;
        let accr = accr_generator(time); // 加速度の入力信号生成

        // 実モデルのシミュレーション
        let state_true = model_true.sim_nextstate(vec![accr + inputdist_pos.sample(&mut rng)], vec![inputdist_pos.sample(&mut rng), 0.0]).unwrap();
        //let state_true = model_true.sim_nextstate(vec![accr], vec![0.0, 0.0]).unwrap();
        x_true.recordvalue(state_true[0]).unwrap();
        v_true.recordvalue(state_true[1]).unwrap();

        // 観測
        let v_obs = model_true.observation(vec![spdsensdist.sample(&mut rng)]).unwrap();

        // 車輪速の積分によるデッドレコニング
        let state_odo = model_odo.predict_nextstate(vec![v_obs[0]]).unwrap();
        x_odo.recordvalue(state_odo[0]).unwrap();
        v_odo.recordvalue(state_odo[1]).unwrap();

        // カルマンフィルタによる推定
        let state_predict = model_est.predict_nextstate(vec![accr]);
        let state_update = model_est.update_nextstate(v_obs).unwrap();
        x_est.recordvalue(state_update[0]).unwrap();
        v_est.recordvalue(state_update[1]).unwrap();
    }

    let last_step = SIMULATION_STEPS as usize - 1;
    println!("x_true = {}, x_est = {}, x_odo = {}", &x_true.getvalue_bystep(last_step).unwrap(), &x_est.getvalue_bystep(last_step).unwrap(), &x_odo.getvalue_bystep(last_step).unwrap());
    
    // 計測データのプロット
    let plots_x = vec![x_true, x_est, x_odo];
    let plots_v = vec![v_true, v_est, v_odo];
    timeplot(&plots_x, "plots_x.png", "plots_x");
    timeplot(&plots_v, "plots_v.png", "plots_v");

    
}

fn create_model2(sample_time: f64) -> Model {
    let mut model = Model::new(2, 1, 1);
    
    model.init_state(vec![0.0, 0.0], vec![0.01, 0.0, 0.0, 0.001]).unwrap();
    model.set_mat_a(vec![1.0, sample_time, 0.0, 1.0]).unwrap();
    model.set_mat_b(vec![0.0, sample_time]).unwrap();
    model.set_mat_c(vec![0.0, 1.0]).unwrap();
    model.set_mat_q(vec![0.1, 0.0, 0.0, 0.1]).unwrap();
    model.set_mat_r(vec![0.1]).unwrap();

    model
}


fn create_model_odo(sample_time: f64) -> Model { 
    let mut model = Model::new(2, 1, 1);
    // 車輪速から位置を積分するモデル
    model.init_state(vec![0.0, 0.0], vec![0.1, 0.0, 0.0, 0.1]).unwrap();
    model.set_mat_a(vec![1.0, sample_time, 0.0, 0.0]).unwrap();
    model.set_mat_b(vec![0.0, 1.0]).unwrap();
    model.set_mat_c(vec![0.0, 1.0]).unwrap();
    model.set_mat_q(vec![0.01, 0.0, 0.0, 0.001]).unwrap();
    model.set_mat_r(vec![0.05]).unwrap();

    model
}

fn accr_generator(simtime: f64) -> f64{
    let time = simtime - ((simtime / 10.0) as i32) as f64 * 10.0;

    if time < 1.0 {
        return 0.0;
    }
    else if time < 3.0 {
        return 1.0;
    }
    else if time < 5.0 {
        return 0.0;
    }
    else if time < 7.0 {
        return -1.0;
    }
    else {
        return 0.0;
    }
}