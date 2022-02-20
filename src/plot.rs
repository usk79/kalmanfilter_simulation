use plotters::prelude::*;

use std::fs;
use std::io::{BufWriter, Write};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Point {
    pub x : f64,
    pub y : f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Point{ x: x, y: y}
    }
}

#[derive(Debug)]
pub struct Points {
    pub data: Vec<Point>,   // データ（時間と値のベクタ） いったんpub 本来はPrivateにしたい
    pub length: usize,      // データ長
}

impl Points {
    pub fn new(len: usize) -> Self {
        Points {
            data: vec![Point::new(0.0, 0.0); len],
            length: len,
        }
    }
}

trait Series {
    fn xmax(&self) -> f64;
    fn xmin(&self) -> f64;
    fn ymax(&self) -> f64;
    fn ymin(&self) -> f64;
    fn calc_minmax(&self) -> (f64, f64, f64, f64);
}

impl Series for Points {
    fn xmax(&self) -> f64 {
        self.calc_minmax().0
    }

    fn xmin(&self) -> f64 {
        self.calc_minmax().1
    }

    fn ymax(&self) -> f64 {
        self.calc_minmax().2
    }

    fn ymin(&self) -> f64 {
        self.calc_minmax().3
    }

    fn calc_minmax(&self) -> (f64, f64, f64, f64) {
        let mut xmin = self.data[0].x;
        let mut xmax = self.data[0].x;
        let mut ymin = self.data[0].y;
        let mut ymax = self.data[0].y;

        for pt in &self.data {
            if pt.x < xmin { xmin = pt.x; }
            else if pt.x > xmax { xmax = pt.x; }
            
            if pt.y < ymin { ymin = pt.y; }
            else if pt.y > ymax { ymax = pt.y; }
        }

        (xmin, xmax, ymin, ymax)
    }
    
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct TimeValue {
    pub t : f64,
    pub v : f64,
}

impl TimeValue {
    pub fn new(initvalue: f64) -> Self {
        TimeValue {t: 0.0, v: initvalue}
    }
}

#[derive(Debug)]
pub struct TimeSeries {
    data: Vec<TimeValue>,   // データ（時間と値のベクタ）
    ctrlcycle: f64,         // 制御周期[s]
    pub length: usize,          // データ長
    step: usize,            // 現在のステップ
}

impl TimeSeries {
    pub fn new(cycle: f64, simtime: f64, initvalue: f64) -> Self {
        let len = (simtime / cycle).ceil() as usize;

        TimeSeries {
            data: vec![TimeValue::new(initvalue); len],
            ctrlcycle: cycle,
            length: len,
            step: 0,
        }
    }

    pub fn recordvalue(&mut self, value: f64) -> Result<(), &str> { // 値を記録して1ステップ進める
        if self.step >= self.length {
            return Err("err");
        }
        let timenow = self.data[self.step].t + self.ctrlcycle;

        self.step += 1;
        self.data[self.step].t = timenow;
        self.data[self.step].v = value;
        Ok(())
    }

    pub fn getvalue_bystep(&self, step: usize) -> Result<f64, &str> {
        if self.step >= self.length {
            return Err("err");
        }
        return Ok(self.data[step].v);
    }

    pub fn gettime_bystep(&self, step: usize) -> Result<f64, &str> {
        if self.step >= self.length {
            return Err("err");
        }
        
        Ok(self.data[step].t)
    }

    pub fn output_csv(&self, filename: &str) {
        
        let mut f = BufWriter::new(fs::File::create(filename).unwrap());

        for elem in &self.data {
            let line_str = format!("{}, {}\n", elem.t, elem.v);
            f.write(line_str.as_bytes()).unwrap();
        }

    }

}

impl Series for TimeSeries {
    fn xmax(&self) -> f64 {
        self.data[0].t
    }

    fn xmin(&self) -> f64 {
        self.data[self.length - 1].t
    }

    fn ymax(&self) -> f64 {
        self.calc_minmax().2
    }

    fn ymin(&self) -> f64 {
        self.calc_minmax().3
    }

    fn calc_minmax(&self) -> (f64, f64, f64, f64) {
        let mut ymin = self.data[0].v;
        let mut ymax = self.data[0].v;

        for pt in &self.data {           
            if pt.v < ymin { ymin = pt.v; }
            else if pt.v > ymax { ymax = pt.v; }
        }

        (self.data[0].t, self.data[self.length - 1].t, ymin, ymax)
    }

}


const PLT_WIDTH: u32 = 500; // プロット幅 [px]
const PLT_HEIGHT: u32 = 500; // プロット高さ [px]

pub fn scatter(series : &Points, filename : &str, caption : &str) {

    let plt = BitMapBackend::new(filename, (PLT_WIDTH, PLT_HEIGHT)).into_drawing_area();
    plt.fill(&WHITE).unwrap();

    let font = ("sans-serif", 20);

    let minmax = series.calc_minmax();
    let xrange = minmax.0.floor()..minmax.1.ceil();
    let yrange = minmax.2.floor()..minmax.3.ceil();
  
    let mut chart = ChartBuilder::on(&plt)
      .caption(caption, font.into_font()) // キャプションのフォントやサイズ
      .margin(10)                         // 上下左右全ての余白
      .x_label_area_size(16)              // x軸ラベル部分の余白
      .y_label_area_size(42)              // y軸ラベル部分の余白
      .build_cartesian_2d(                // x軸とy軸の数値の範囲を指定する
        xrange,                           // x軸の範囲
        yrange)                           // y軸の範囲
      .unwrap();

    // x軸y軸、グリッド線などを描画
    chart.configure_mesh().draw().unwrap();

    let line_series = PointSeries::<_,_,Circle<_,_>,_>::new(
        series.data.iter().map(|pt| (pt.x, pt.y)),
        1, &RED);

    chart.draw_series(line_series).unwrap();
}

// 上のscatterと共通化したい　Iteratorを理解すればできるはず
pub fn timeplot(plots : &Vec<TimeSeries>, filename : &str, caption : &str) {

    let plt = BitMapBackend::new(filename, (PLT_WIDTH, PLT_HEIGHT)).into_drawing_area();
    plt.fill(&WHITE).unwrap();

    let font = ("sans-serif", 20);

    let minmax = plots[0].calc_minmax();
    let xrange = minmax.0.floor()..minmax.1.ceil();
    let yrange = minmax.2.floor()..minmax.3.ceil();
  
    let mut chart = ChartBuilder::on(&plt)
      .caption(caption, font.into_font()) // キャプションのフォントやサイズ
      .margin(10)                         // 上下左右全ての余白
      .x_label_area_size(16)              // x軸ラベル部分の余白
      .y_label_area_size(42)              // y軸ラベル部分の余白
      .build_cartesian_2d(                // x軸とy軸の数値の範囲を指定する
        xrange,                           // x軸の範囲
        yrange)                           // y軸の範囲
      .unwrap();

    // x軸y軸、グリッド線などを描画
    chart.configure_mesh().draw().unwrap();

    let colorlist = vec![&RED, &BLUE, &GREEN, &CYAN, &MAGENTA, &YELLOW];
    let mut idx = 0;
    for plot in plots { 
        
    
        let line_series = PointSeries::<_,_,Circle<_,_>,_>::new(
            plot.data.iter().map(|pt| (pt.t, pt.v)),
            1, colorlist[idx % colorlist.len()]);

        chart.draw_series(line_series).unwrap();
        idx += 1;
    }

}

pub fn output_timeseries(datalist: &Vec<TimeSeries>, filename: &String) {
    let b = b"test";

    let mut f = BufWriter::new(fs::File::create(filename).unwrap());
    for _ in 0 .. 100_000_000 {
        f.write(b).unwrap();
    }
}