/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.gitia.clase2017.rnn;

import java.util.Random;
import org.ejml.simple.SimpleMatrix;
import org.gitia.froog.Feedforward;
import org.gitia.froog.layer.Layer;
import org.gitia.froog.lossfunction.LossFunction;
import org.gitia.froog.trainingalgorithm.Backpropagation;
import org.gitia.froog.transferfunction.TransferFunction;

/**
 *
 * @author Matías Rodschild <mroodschild@gmail.com>
 */
public class ClasificacionXOR {

    public static void main(String[] args) {
        double[][] x = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}};
        double[][] t = {
            {-1},
            {1},
            {1},
            {-1}};
        SimpleMatrix X = new SimpleMatrix(x);
        SimpleMatrix T = new SimpleMatrix(t);

        Random seed = new Random(1);
        Feedforward net = new Feedforward();
        net.addLayer(new Layer(2, 3, TransferFunction.TANSIG, seed));
        net.addLayer(new Layer(3, 1, TransferFunction.TANSIG, seed));

        Backpropagation bp = new Backpropagation();
        bp.setEpoch(200);
        bp.setMomentum(0.9);
        bp.setLearningRate(0.01);
        bp.setLossFunction(LossFunction.MSE);

        bp.train(net, X.transpose(), T.transpose());

        net.output(X.transpose()).print();
    }
}
