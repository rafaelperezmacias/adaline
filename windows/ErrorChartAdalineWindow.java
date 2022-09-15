package windows;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.block.BlockBorder;
import org.jfree.chart.labels.StandardXYToolTipGenerator;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.title.TextTitle;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.util.ShapeUtilities;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;

public class ErrorChartAdalineWindow extends JFrame {

    private XYSeries errorSeries;
    private ArrayList<ValueForSeries> seriesValues;

    public ErrorChartAdalineWindow(AdalineWindow adalineWindow) {
        setTitle("Error cuadratico por epoca del Adaline");
        setSize(600,400);
        int xFather = adalineWindow.getX();
        int yFather = adalineWindow.getY();
        setLocation(xFather + adalineWindow.getWidth() - 600, yFather);
        seriesValues = new ArrayList<>();
        // Serie, la que contiene los valores a imprimirse
        errorSeries = new XYSeries("Error cuadratico");
        // Conjunto de datos al cual agregaremos la serie
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(errorSeries);
        // Creamos la grafica con el conjunto de datos
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Error cuadratico por epoca en Adaline",
                "Error",
                "Epoch",
                dataset,
                PlotOrientation.HORIZONTAL,
                true,
                true,
                false
        );
        // Obtenemos el ploteo para poder manipular la forma en la que se pintan
        XYPlot plot = chart.getXYPlot();
        // Creamos un nuevo render de linea para configurarlo nosotros.
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesPaint(0, Color.RED);
        renderer.setSeriesStroke(0, new BasicStroke(2.0f));
        renderer.setSeriesShapesVisible(0, true);
        renderer.setSeriesShape(0, ShapeUtilities.createDiamond(1.0f));
        StandardXYToolTipGenerator tooltipGenerator = new StandardXYToolTipGenerator()
        {
            @Override
            public String generateToolTip(XYDataset dataset, int series, int item)
            {
                if ( errorSeries.getItems().size() != 0 && seriesValues.size() != 0 ) {
                    return "Valor: " + dataset.getXValue(series, item);
                }
                return "";
            }
        };
        renderer.setBaseToolTipGenerator(tooltipGenerator);
        // Lo agregamos en nuestro grafico
        plot.setRenderer(renderer);
        // Fondo blanco
        plot.setBackgroundPaint(Color.white);
        // Marca de rendijas (y)
        plot.setRangeGridlinesVisible(true);
        plot.setRangeGridlinePaint(Color.BLACK);
        // Marca de rendijas (x)
        plot.setDomainGridlinesVisible(true);
        plot.setDomainGridlinePaint(Color.BLACK);
        // Quitamos los bordes
        chart.getLegend().setBorder(BlockBorder.NONE);
        //
        chart.setTitle(new TextTitle("Error cuadratico por epoca en Adaline",
                        new Font("Dialog", java.awt.Font.BOLD, 18)
                )
        );
        // Acomodo del grafico dentro del JFrame
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setBorder(BorderFactory.createEmptyBorder(15, 15, 15, 15));
        chartPanel.setBackground(Color.white);
        add(chartPanel);

        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        setVisible(true);
    }

    public void addValueForSeries(int epoch, double error) {
        seriesValues.add(new ValueForSeries(epoch, error));
        errorSeries.clear();
        for ( int i = seriesValues.size() - 1; i >= 0; i-- ) {
            ValueForSeries value = seriesValues.get(i);
            errorSeries.add(value.error, value.epoch);
        }
    }

    private static class ValueForSeries {

        public int epoch;
        public double error;

        public ValueForSeries(int epoch, double error) {
            this.epoch = epoch;
            this.error = error;
        }

    }

}

