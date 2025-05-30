#![feature(generic_const_exprs)]
#![feature(trivial_bounds)]
#![feature(generic_arg_infer)]
#![allow(incomplete_features)]

pub mod plot;
use crate::plot::{plot_comparison,loss_curve};

use slut::{dimension::{self, Dimensionless, *}, dless, dot, tensor::*,units::{self, Unitless}};

const N: usize = 7;

fn fac(n: i32) -> i32 {

    let mut r = 1;
    for i in 1..n+1 {
        r*= i;
    }

    r
}

fn infer(coeffs: Vector<f64,Dimensionless,N>,x: f64) -> f64 {

    // create a vector like [1,x¹,x²,... x^N]
    let inputs =  Vector::<f64,Dimensionless,N>::init_2d(
        |i,_| x.powf((i) as f64)
    );

    let y = dot!(coeffs,inputs);

    y.raw()
}

fn main() {

    println!("{}",fac(3));

    let target = |a:f64| a;

    const h: f64 = 5e-15;
    const lr: f64 = 5e-8;

    let loss = |r:f64,t:f64| (r-t).powi(2);

    let mut coeffs = Vector::<f64,Dimensionless,N>::zero();
    println!("{}",coeffs);

    let step = 0.01;
    let max = 5.0;

    // Compute MSE over all training points
    let mse = |c: &Vector<f64,Dimensionless,N>| {
        let mut total_loss = 0.0;
        for i in 0..((max/step) as i32) {
            let x = i as f64 * step;
            total_loss += loss(infer(c.clone(), x), target(x));
        }
        total_loss / ((max/step) as f64)
    };

    // Gradient using numerical differentiation of MSE
    let grad = |c: &Vector<f64,Dimensionless,N>, k: usize| {
        let perturbation = Vector::<f64,Dimensionless,N>::init_2d(|i,j| if i+j == k {h} else {0.0});
        (mse(&(c.clone() + perturbation)) - mse(c)) / h
    };

    let epochs = 10000;

    println!("Step: {}, Max: {}, Epochs: {}", step, max, epochs);
    println!("Target: {}", target(1.0));

    let stating_loss = mse(&coeffs);
    println!("Starting loss: {}", stating_loss);

    println!("Coeffs: {}", coeffs);

    let mut l = 0.0;
    let mut losses = Vec::<f64>::new();

    for e in  0..epochs {

        let mut g_norm = 0.0;
        for k in 0..N {
            let g = grad(&coeffs, k);
            g_norm += g * g;
            coeffs.set_at(0,k, 0, coeffs.get_at(0,k,0) - dless!(g * lr));
        }
        g_norm = g_norm.sqrt();
        println!("Gradient Norm: {}", g_norm);
        
        // compute the new coef
        println!("Coeffs: {}", coeffs);

        // compute the loss using the MSE function
        l = mse(&coeffs);
        losses.push(l);

        println!("Epoch: {}, Loss: {}", e, l);
    }

    // show a sampled version of the losses
    loss_curve(&losses.iter().map(|&x| x).step_by((1) as usize).collect::<Vec<_>>(),  "loss_curve.html")
        .expect("Failed to create loss curve visualization");

    println!("Final Coeffs: {}", coeffs);
    println!("Final Loss: {}", l);
    println!("Starting Loss: {}", stating_loss);

    let f = |x: f64| infer(coeffs.clone(), x);

    println!("f(1.0) = {}", f(1.0));
    println!("f(1.5) = {}", f(1.5));
    println!("f(2.0) = {}", f(2.0));
    println!("f(3.0) = {}", f(3.0));

    println!("Target(1.0) = {}", target(1.0));
    println!("Target(1.5) = {}", target(1.5));
    println!("Target(2.0) = {}", target(2.0));
    println!("Target(3.0) = {}", target(3.0));

    // Visualization code
    let output_file = "visualization.html";
    plot_comparison(f, target, 0.0, 5.0, 500, output_file)
        .expect("Failed to create visualization");

}
