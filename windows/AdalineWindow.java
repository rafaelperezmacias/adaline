package windows;

import models.Adaline;
import utils.DataSet;

import javax.swing.*;
import javax.swing.event.CaretEvent;
import javax.swing.event.CaretListener;
import java.awt.*;
import java.awt.event.*;
import java.awt.geom.Line2D;
import java.util.ArrayList;
import java.util.Arrays;

public class AdalineWindow extends JFrame {

    private ArrayList<Point> points;
    private PerceptronThread.Model pModel;
    private AdalineThread.Model aModel;

    private final double[] defaultWeights = { 0.0, 0.0, 0.0 };

    private final int MAP_WIDTH = 500;
    private final int MAP_HEIGHT = 500;
    private final int RADIUS_POINT = 5;
    private final double MAP_SCALE = 5.0;

    private final Color LEFTCLICK_COLOR = Color.blue;
    private final Color RIGHTCLICK_COLOR = Color.green;

    private double[] weights;
    private double[] perceptronWeights;
    private double[] adalineWeights;

    private final Map map;

    private JLabel lblPEpochResult;
    private JLabel lblPWeightResult0;
    private JLabel lblPWeightResult1;
    private JLabel lblPWeightResult2;

    private JLabel lblAEpochResult;
    private JLabel lblAWeightResult0;
    private JLabel lblAWeightResult1;
    private JLabel lblAWeightResult2;
    private JLabel lblAErrorResult;

    private JTextField txtWeight0;
    private JTextField txtWeight1;
    private JTextField txtWeight2;

    private JTextField txtLearningRate;
    private JTextField txtEpochs;
    private JTextField txtMinError;

    private JButton btnRandomWeights;
    private JButton btnPerceptron;
    private JButton btnAdaline;

    private boolean clickEnable;
    private boolean adalineModelEnable;
    private boolean perceptronModelEnable;
    private boolean changeWeights;
    private boolean addInstanceEnable;
    private boolean selectedAdaline;
    private boolean selectedPerceptron;

    private JMenu jmOptions;
    private JMenu jmPredict;
    private JMenu jmAdaline;
    private JMenu jmPerceptron;
    private JRadioButtonMenuItem jmiPerceptronPredict;
    private JRadioButtonMenuItem jmiAdalinePredict;

    private final String unicodeSubscript0 = "\u2080";
    private final String unicodeSubscript1 = "\u2081";
    private final String unicodeSubscript2 = "\u2082";

    private ErrorChartAdalineWindow errorChartAdalineWindow;

    public AdalineWindow()
    {
        super("Ejemplo de funcionamiento del adaline");
        setLayout(null);
        setSize(1200,625);
        setLocationRelativeTo(null);
        // Inicializamos la lista que contiene los puntos del mapa
        points = new ArrayList<>();
        // Inicializamos los pesos con lo que vamos a trabajar
        perceptronWeights = new double[3];
        adalineWeights = new double[3];
        weights = new double[3];
        System.arraycopy(defaultWeights, 0, weights, 0, defaultWeights.length);
        System.arraycopy(defaultWeights, 0, perceptronWeights, 0, defaultWeights.length);
        System.arraycopy(defaultWeights, 0, adalineWeights, 0, defaultWeights.length);
        // Barra de menu
        JMenuBar menuBar = new JMenuBar();
        setJMenuBar(menuBar);
        // Un menu de la barra
        jmOptions = new JMenu("Opciones");
        menuBar.add(jmOptions);
        // Opciones del menu
        // ELiminar ultima
        JMenuItem jmiDeleteLastInstance = new JMenuItem("Eliminar la ultima instancia");
        jmiDeleteLastInstance.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if ( !points.isEmpty() ) {
                    int idxPoint = -1;
                    for ( int i = points.size() - 1; i >= 0; i-- ) {
                        if ( !points.get(i).sweep ) {
                            idxPoint = i;
                            break;
                        }
                    }
                    if ( idxPoint != -1 ) {
                        points.remove(idxPoint);
                        map.repaint();
                    }
                }
            }
        });
        jmOptions.add(jmiDeleteLastInstance);
        // Limpiar instancias
        JMenuItem jmiClearInstances = new JMenuItem("Limpiar instancias");
        jmOptions.add(jmiClearInstances);
        // Limpiar el programa
        JMenuItem jmiClearAll = new JMenuItem("Limpiar todo");
        jmiClearAll.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                points.clear();
                System.arraycopy(defaultWeights, 0, weights, 0, weights.length);
                System.arraycopy(defaultWeights, 0, adalineWeights, 0, weights.length);
                System.arraycopy(defaultWeights, 0, perceptronWeights, 0, weights.length);
                map.repaint();
                lblPWeightResult0.setText("<html>w" + unicodeSubscript0 + " = <b>" + perceptronWeights[0] + "</b></html>");
                lblPWeightResult1.setText("<html>w" + unicodeSubscript1 + " = <b>" + perceptronWeights[1] + "</b></html>");
                lblPWeightResult2.setText("<html>w" + unicodeSubscript2 + " = <b>" + perceptronWeights[2] + "</b></html>");
                lblAWeightResult0.setText("<html>w" + unicodeSubscript0 + " = <b>" + adalineWeights[0] + "</b></html>");
                lblAWeightResult1.setText("<html>w" + unicodeSubscript1 + " = <b>" + adalineWeights[1] + "</b></html>");
                lblAWeightResult2.setText("<html>w" + unicodeSubscript2 + " = <b>" + adalineWeights[2] + "</b></html>");
                txtWeight0.setText(String.valueOf(weights[0]));
                txtWeight1.setText(String.valueOf(weights[1]));
                txtWeight2.setText(String.valueOf(weights[2]));
                txtLearningRate.setText("0.");
                txtEpochs.setText("");
                txtMinError.setText("");
                lblPEpochResult.setText("<html>Epoca: <b>0</b></html>");
                lblAEpochResult.setText("<html>Epoca: <b>0</b></html>");
                lblAErrorResult.setText("<html>Error: <b>0.0</b></html>");
                addInstanceEnable = true;
                jmPredict.setVisible(false);
                jmAdaline.setVisible(false);
                jmPerceptron.setVisible(false);
            }
        });
        jmOptions.add(jmiClearAll);
        // Salir
        JMenuItem jmiClose = new JMenuItem("Salir");
        jmiClose.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                AdalineWindow.this.dispose();
            }
        });
        jmOptions.add(jmiClose);
        // Opciones para predecir
        jmPredict = new JMenu("Modelo");
        jmPredict.setVisible(false);
        menuBar.add(jmPredict);
        ButtonGroup bgPredict = new ButtonGroup();
        /** Perceptron */
        jmPerceptron = new JMenu("Perceptron");
        jmPerceptron.setVisible(false);
        jmPredict.add(jmPerceptron);
        // Predecir una instancia
        jmiPerceptronPredict = new JRadioButtonMenuItem("Predecir", true);
        bgPredict.add(jmiPerceptronPredict);
        jmPerceptron.add(jmiPerceptronPredict);
        jmiPerceptronPredict.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if ( !changeWeights ) {
                    addInstanceEnable = false;
                    selectedPerceptron = true;
                    selectedAdaline = false;
                }
            }
        });
        // Mostrar barrido
        JMenuItem jmiPerceptronShowSweep = new JMenuItem("Mostrar barrido");
        jmPerceptron.add(jmiPerceptronShowSweep);
        jmiPerceptronShowSweep.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if ( !changeWeights ) {
                    ArrayList<Point> newPoints = new ArrayList<>();
                    for ( Point point : points ) {
                        if ( !point.sweep ) {
                            newPoints.add(point);
                        }
                    }
                    points = newPoints;
                    newPoints = new ArrayList<>();
                    for ( int i = 0; i <= map.getWidth(); i+= RADIUS_POINT ) {
                        for ( int j = 0; j <= map.getHeight(); j+= RADIUS_POINT ) {
                            Point point = new Point();
                            point.xMap = i;
                            point.yMap = j;
                            point.x = ( i >= MAP_WIDTH * 0.5 ) ? i - ( MAP_WIDTH * 0.5 ) : -((MAP_WIDTH * 0.5) - i);
                            point.x /= (MAP_WIDTH * 0.5) / MAP_SCALE;
                            point.y = ( j > MAP_HEIGHT * 0.5 ) ? -(j - (MAP_HEIGHT * 0.5)) : (MAP_HEIGHT * 0.5) - j;
                            point.y /= (MAP_HEIGHT * 0.5) / MAP_SCALE;
                            Object[] instance = new Object[2];
                            instance[0] = point.x;
                            instance[1] = point.y;
                            double result = 0;
                            try {
                                result = pModel.predict(instance);
                                point.leftClick = result == 0.0;
                                point.sweep = true;
                                point.color = result == 0.0 ? LEFTCLICK_COLOR : RIGHTCLICK_COLOR;
                                point.adaline = false;
                                newPoints.add(point);
                            } catch (Exception ex) {
                                System.out.println("No se pudo realizar el barrido");
                                break;
                            }
                        }
                    }
                    newPoints.addAll(points);
                    points = newPoints;
                    map.repaint();
                    if ( selectedAdaline ) {
                        selectedAdaline = false;
                        selectedPerceptron = true;
                        jmiPerceptronPredict.setSelected(true);
                    }
                }
            }
        });
        // Eliminar barrido
        JMenuItem jmiPerceptronHideSweep = new JMenuItem("Ocultar barrido");
        jmPerceptron.add(jmiPerceptronHideSweep);
        jmiPerceptronHideSweep.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if ( !changeWeights && selectedPerceptron ) {
                    ArrayList<Point> newPoints = new ArrayList<>();
                    for ( Point point : points ) {
                        if ( !point.sweep ) {
                            newPoints.add(point);
                        }
                    }
                    points = newPoints;
                    map.repaint();
                }
            }
        });
        /** Adaline */
        jmAdaline = new JMenu("Adaline");
        jmAdaline.setVisible(false);
        jmPredict.add(jmAdaline);
        // Predecir una instancia
        jmiAdalinePredict = new JRadioButtonMenuItem("Predecir", true);
        bgPredict.add(jmiAdalinePredict);
        jmAdaline.add(jmiAdalinePredict);
        jmiAdalinePredict.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if ( !changeWeights ) {
                    addInstanceEnable = false;
                    selectedPerceptron = false;
                    selectedAdaline = true;
                }
            }
        });
        // Mostrar barrido
        JMenuItem jmiAdalineShowSweep = new JMenuItem("Mostrar barrido");
        jmAdaline.add(jmiAdalineShowSweep);
        jmiAdalineShowSweep.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if ( !changeWeights ) {
                    ArrayList<Point> newPoints = new ArrayList<>();
                    for ( Point point : points ) {
                        if ( !point.sweep ) {
                            newPoints.add(point);
                        }
                    }
                    points = newPoints;
                    newPoints = new ArrayList<>();
                    for ( int i = 0; i < map.getWidth(); i+= 1 ) {
                        for ( int j = 0; j < map.getHeight(); j+= 1 ) {
                            Point point = new Point();
                            point.xMap = i;
                            point.yMap = j;
                            point.x = ( i >= MAP_WIDTH * 0.5 ) ? i - ( MAP_WIDTH * 0.5 ) : -((MAP_WIDTH * 0.5) - i);
                            point.x /= (MAP_WIDTH * 0.5) / MAP_SCALE;
                            point.y = ( j > MAP_HEIGHT * 0.5 ) ? -(j - (MAP_HEIGHT * 0.5)) : (MAP_HEIGHT * 0.5) - j;
                            point.y /= (MAP_HEIGHT * 0.5) / MAP_SCALE;
                            Object[] instance = new Object[2];
                            instance[0] = point.x;
                            instance[1] = point.y;
                            double result = 0;
                            try {
                                result = aModel.predict(instance);
                                point.leftClick = result <= 0.5;
                                point.sweep = true;
                                point.adaline = true;
                                if ( result > 0.5 ) {
                                    int intensity = (int) ( ((result - 0.5) / 0.5) * 255);
                                    point.color = new Color(0, 255, 0, intensity);
                                } else {
                                    int intensity = (int) ( (result / 0.5 ) * 255 );
                                    point.color = new Color(0, 0, 255, 255 - intensity);
                                }
                                newPoints.add(point);
                            } catch (Exception ex) {
                                System.out.println("No se pudo realizar el barrido");
                                break;
                            }
                        }
                    }
                    newPoints.addAll(points);
                    points = newPoints;
                    if ( selectedPerceptron ) {
                        selectedPerceptron = false;
                        selectedAdaline = true;
                        jmiAdalinePredict.setSelected(true);
                    }
                    map.repaint();
                }
            }
        });
        // Eliminar barrido
        JMenuItem jmiAdalineHideSweep = new JMenuItem("Ocultar barrido");
        jmAdaline.add(jmiAdalineHideSweep);
        jmiAdalineHideSweep.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if ( !changeWeights && selectedAdaline ) {
                    ArrayList<Point> newPoints = new ArrayList<>();
                    for ( Point point : points ) {
                        if ( !point.sweep ) {
                            newPoints.add(point);
                        }
                    }
                    points = newPoints;
                    map.repaint();
                }
            }
        });
        /** Instancia */
        // Agregar una nueva instancia
        JRadioButtonMenuItem jmiNewInstance = new JRadioButtonMenuItem("Nueva instancia");
        bgPredict.add(jmiNewInstance);
        jmPredict.add(jmiNewInstance);
        jmiNewInstance.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                addInstanceEnable = true;
            }
        });
        // Limpiar instancias
        jmiClearInstances.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                ArrayList<Point> newPoints = new ArrayList<>();
                for ( Point point : points ) {
                    if ( point.sweep ) {
                        newPoints.add(point);
                    }
                }
                points = newPoints;
                map.repaint();
                addInstanceEnable = true;
                selectedPerceptron = false;
                selectedAdaline = false;
                jmiNewInstance.setSelected(true);
            }
        });
        // Lienzo princiapal de la ventana
        map = new Map();
        map.setSize(MAP_WIDTH, MAP_HEIGHT);
        map.setLocation(35,30);
        map.setBackground(Color.WHITE);
        add(map);
        // Eventos del mouse
        clickEnable = true;
        perceptronModelEnable = false;
        adalineModelEnable = false;
        addInstanceEnable = true;
        map.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Point point = new Point();
                // Tratamiento de los datos
                point.xMap = e.getX();
                point.yMap = e.getY();
                point.x = ( e.getX() >= MAP_WIDTH * 0.5 ) ? e.getX() - ( MAP_WIDTH * 0.5 ) : -((MAP_WIDTH * 0.5) - e.getX());
                point.x /= (MAP_WIDTH * 0.5) / MAP_SCALE;
                point.y = ( e.getY() > MAP_HEIGHT * 0.5 ) ? -(e.getY() - (MAP_HEIGHT * 0.5)) : (MAP_HEIGHT * 0.5) - e.getY();
                point.y /= (MAP_HEIGHT * 0.5) / MAP_SCALE;
                // Prediccion de la nueva instancia;
                if ( !addInstanceEnable && (perceptronModelEnable || adalineModelEnable) && e.getButton() == MouseEvent.BUTTON1 ) {
                    Object[] instance = new Object[2];
                    instance[0] = point.x;
                    instance[1] = point.y;
                    try {
                        double result = -1;
                        if ( adalineModelEnable && selectedAdaline ) {
                            result = aModel.predict(instance);
                            JOptionPane.showMessageDialog(null, "La nueva instancia es: " + ((result <= 0.5) ? "circulo azul" : "cuadrado verde"), "Resultado", JOptionPane.INFORMATION_MESSAGE);
                            point.leftClick = result <= 0.5;
                            point.color = result <= 0.5 ? LEFTCLICK_COLOR : RIGHTCLICK_COLOR;
                            point.sweep = false;
                        } else if ( perceptronModelEnable && selectedPerceptron ) {
                            result = pModel.predict(instance);
                            JOptionPane.showMessageDialog(null, "La nueva instancia es: " + ((result == 0.0) ? "circulo azul" : "cuadrado verde"), "Resultado", JOptionPane.INFORMATION_MESSAGE);
                            point.leftClick = result == 0.0;
                            point.color = result == 0.0 ? LEFTCLICK_COLOR : RIGHTCLICK_COLOR;
                            point.sweep = false;
                        } else {
                            System.out.println("No se pudo predecir la instancia");
                            return;
                        }
                        points.add(point);
                        map.repaint();
                    } catch (Exception ex) {
                        System.out.println("No se pudo predecir la instancia");
                    }
                    return;
                }
                // Validacion para agregar un nuevo elemento
                if ( (e.getButton() != MouseEvent.BUTTON1 && e.getButton() != MouseEvent.BUTTON3) || !clickEnable || !addInstanceEnable ) {
                    return;
                }
                point.sweep = false;
                // Boton izquierdo
                if ( e.getButton() == MouseEvent.BUTTON1 ) {
                    point.leftClick = true;
                    point.color = LEFTCLICK_COLOR;
                    points.add(point);
                }
                // Boton derecho
                if ( e.getButton() == MouseEvent.BUTTON3 ) {
                    point.leftClick = false;
                    point.color = RIGHTCLICK_COLOR;
                    points.add(point);
                }
                System.out.println("Nuevo punto agregado: " + point);
                jmPredict.setVisible(false);
                if ( adalineModelEnable || perceptronModelEnable ) {
                    adalineModelEnable = false;
                    perceptronModelEnable = false;
                    ArrayList<Point> newPoints = new ArrayList<>();
                    for ( Point tmpPoint : points ) {
                        if ( !tmpPoint.sweep ) {
                            newPoints.add(tmpPoint);
                        }
                    }
                    points = newPoints;
                }
                map.repaint();
            }
        });
        /** Titulos, leyendas */
        // Escala del norte del plano
        JLabel lblScaleNorth = new JLabel("+ " + MAP_SCALE);
        lblScaleNorth.setSize(28,10);
        lblScaleNorth.setLocation(map.getX() + ( map.getWidth() / 2 ) - 12, map.getY() - 15);
        add(lblScaleNorth);
        // Escala del sur del plano
        JLabel lblScaleSouth = new JLabel("- " + MAP_SCALE);
        lblScaleSouth.setSize(28,10);
        lblScaleSouth.setLocation(map.getX() + ( map.getWidth() / 2 ) - 12, map.getY() + map.getHeight() + 5);
        add(lblScaleSouth);
        // Escala del este del plano
        JLabel lblScaleEast = new JLabel("+ " + MAP_SCALE);
        lblScaleEast.setSize(28,10);
        lblScaleEast.setLocation(map.getX() + ( map.getWidth() ) + 5, map.getY() + (map.getHeight() / 2) - 5);
        add(lblScaleEast);
        // Escala del este del plano
        JLabel lblScaleWest = new JLabel("- " + MAP_SCALE);
        lblScaleWest.setSize(28,10);
        lblScaleWest.setLocation(map.getX() - 27, map.getY() + (map.getHeight() / 2) - 5);
        add(lblScaleWest);
        // Subtitulo de la ventana
        JLabel lblSubtitle = new JLabel("Configuracion de los parametros");
        lblSubtitle.setLocation(map.getX() + map.getWidth() + 40, map.getY());
        lblSubtitle.setSize(getWidth() - (map.getX() + map.getWidth() + 75), 24);
        lblSubtitle.setHorizontalAlignment(JLabel.CENTER);
        lblSubtitle.setFont(new Font("Dialog", Font.BOLD, 18));
        add(lblSubtitle);
        // Subtitulo Perceptron & Adaline
        JLabel lblPASubtitle = new JLabel("Perceptron & Adaline");
        lblPASubtitle.setLocation(lblSubtitle.getX(), lblSubtitle.getY() + lblSubtitle.getY() + 10);
        lblPASubtitle.setSize((int) ((getWidth() - (map.getX() + map.getWidth() + 75)) * 0.45), 20);
        lblPASubtitle.setHorizontalAlignment(JLabel.CENTER);
        lblPASubtitle.setFont(new Font("Dialog", Font.PLAIN, 16));
        add(lblPASubtitle);
        // Separador
        JSeparator jsParameters = new JSeparator();
        jsParameters.setOrientation(SwingConstants.VERTICAL);
        jsParameters.setLocation(lblPASubtitle.getX() + (lblSubtitle.getWidth() / 2), lblPASubtitle.getY());
        jsParameters.setSize(2, 130);
        add(jsParameters);
        //Subtitulo para Adaline
        JLabel lblAdalineSubtitule = new JLabel("Adaline");
        lblAdalineSubtitule.setLocation(jsParameters.getX() + (int) ((getWidth() - (map.getX() + map.getWidth() + 75)) * 0.05), lblSubtitle.getY() + 30);
        lblAdalineSubtitule.setSize(lblPASubtitle.getSize());
        lblAdalineSubtitule.setHorizontalAlignment(JLabel.CENTER);
        lblAdalineSubtitule.setFont(new Font("Dialog", Font.PLAIN, 16));
        add(lblAdalineSubtitule);
        // Factor de aprendizaje
        JLabel lblLearningRate = new JLabel("Learning rate: ");
        lblLearningRate.setLocation(lblPASubtitle.getX(), lblPASubtitle.getY() + lblPASubtitle.getHeight() + 10);
        lblLearningRate.setSize((int) (lblPASubtitle.getWidth() * 0.40), 40);
        lblLearningRate.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblLearningRate);
        txtLearningRate = new JTextField("0.");
        txtLearningRate.setLocation(lblLearningRate.getX() + lblLearningRate.getWidth() + ((int) (lblPASubtitle.getWidth() * 0.02)), lblLearningRate.getY());
        txtLearningRate.setSize((int) (lblPASubtitle.getWidth() * 0.58), 40);
        txtLearningRate.addKeyListener(new KeyAdapter() {
            @Override
            public void keyTyped(KeyEvent e) {
                if ( e.getKeyChar() < '0' || e.getKeyChar() > '9' || txtLearningRate.getCaretPosition() < 2 ) {
                    e.consume();
                }
                super.keyTyped(e);
            }
            @Override
            public void keyPressed(KeyEvent e) {
                if ( txtLearningRate.getText().length() == 2 && e.getKeyCode() == KeyEvent.VK_BACK_SPACE ) {
                    e.consume();
                }
                super.keyPressed(e);
            }
        });
        add(txtLearningRate);
        // Epocas
        JLabel lblEpochs = new JLabel("Epocas: ");
        lblEpochs.setSize(lblLearningRate.getSize());
        lblEpochs.setLocation(lblLearningRate.getX(), lblLearningRate.getY() + lblLearningRate.getHeight() + 10);
        lblEpochs.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblEpochs);
        txtEpochs = new JTextField();
        txtEpochs.setSize(txtLearningRate.getSize());
        txtEpochs.setLocation(txtLearningRate.getX(), lblEpochs.getY());
        txtEpochs.addKeyListener(new KeyAdapter() {
            @Override
            public void keyTyped(KeyEvent e) {
                if ( e.getKeyChar() < '0' || e.getKeyChar() > '9' ) {
                    e.consume();
                }
                super.keyTyped(e);
            }
        });
        add(txtEpochs);
        // Error minimo
        JLabel lblError = new JLabel("Error minimo: ");
        lblError.setLocation(lblAdalineSubtitule.getX(), lblLearningRate.getY());
        lblError.setSize((int) (lblAdalineSubtitule.getWidth() * 0.40), 40);
        lblError.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblError);
        txtMinError = new JTextField();
        txtMinError.setLocation(lblError.getX() + lblError.getWidth() + ((int) (lblAdalineSubtitule.getWidth() * 0.02)), lblLearningRate.getY());
        txtMinError.setSize((int) (lblAdalineSubtitule.getWidth() * 0.58), 40);
        txtMinError.addKeyListener(new CustomKeyListener(txtMinError));
        add(txtMinError);
        // Tipo de funcion
        JLabel lblFunction = new JLabel("Funcion: ");
        lblFunction.setLocation(lblAdalineSubtitule.getX(), lblEpochs.getY());
        lblFunction.setSize((int) (lblAdalineSubtitule.getWidth() * 0.40), 40);
        lblFunction.setFont(new Font("Dialog", Font.PLAIN, 14));
        lblFunction.setVisible(false);
        add(lblFunction);
        JComboBox<String> jcbFunction = new JComboBox<>();
        jcbFunction.setLocation(txtMinError.getX(), lblEpochs.getY());
        jcbFunction.setSize((int) (lblAdalineSubtitule.getWidth() * 0.58), 40);
        jcbFunction.setCursor(new Cursor(Cursor.HAND_CURSOR));
        jcbFunction.addItem("Sigmoidal");
        jcbFunction.addItem("Tangente hiperbolica");
        jcbFunction.setVisible(false);
        add(jcbFunction);
        // Título de los pesos
        JLabel lblWeigths = new JLabel("Pesos");
        lblWeigths.setLocation(lblSubtitle.getX(), txtEpochs.getY() + txtEpochs.getHeight() + 20);
        lblWeigths.setSize(lblSubtitle.getSize());
        lblWeigths.setHorizontalAlignment(JLabel.CENTER);
        lblWeigths.setFont(new Font("Dialog", Font.PLAIN, 16));
        add(lblWeigths);
        // Peso 0
        JLabel lblWeigth0 = new JLabel("w" + unicodeSubscript0);
        lblWeigth0.setLocation(lblLearningRate.getX(), lblWeigths.getY() + lblWeigths.getHeight() + 10);
        lblWeigth0.setSize((lblSubtitle.getWidth() / 4), 25);
        lblWeigth0.setHorizontalAlignment(JLabel.CENTER);
        lblWeigth0.setFont(new Font("Dialog", Font.BOLD, 20));
        add(lblWeigth0);
        txtWeight0 = new JTextField();
        txtWeight0.setLocation(lblSubtitle.getX(), lblWeigth0.getY() + lblWeigth0.getHeight() + 5);
        txtWeight0.setSize((lblSubtitle.getWidth() / 4),40);
        txtWeight0.setText(String.valueOf(weights[0]));
        add(txtWeight0);
        // Peso 1
        JLabel lblWeigth1 = new JLabel("w" + unicodeSubscript1);
        lblWeigth1.setLocation(lblSubtitle.getX() + (lblSubtitle.getWidth() / 2) - lblWeigth0.getWidth(), lblWeigth0.getY());
        lblWeigth1.setSize(lblWeigth0.getSize());
        lblWeigth1.setHorizontalAlignment(JLabel.CENTER);
        lblWeigth1.setFont(new Font("Dialog", Font.BOLD, 20));
        add(lblWeigth1);
        txtWeight1 = new JTextField();
        txtWeight1.setLocation(lblWeigth1.getX(), txtWeight0.getY());
        txtWeight1.setSize(txtWeight0.getSize());
        txtWeight1.setText(String.valueOf(weights[1]));
        add(txtWeight1);
        // Peso 2
        txtWeight2 = new JTextField();
        txtWeight2.setLocation(lblSubtitle.getX() + (lblSubtitle.getWidth() / 2) , txtWeight1.getY());
        txtWeight2.setSize(txtWeight0.getWidth(),txtWeight0.getHeight());
        txtWeight2.setText(String.valueOf(weights[2]));
        add(txtWeight2);
        JLabel lblWeigth2 = new JLabel("w" + unicodeSubscript2);
        lblWeigth2.setLocation(txtWeight2.getX(), txtWeight1.getY() - 30);
        lblWeigth2.setSize(txtWeight2.getWidth(), 25);
        lblWeigth2.setHorizontalAlignment(JLabel.CENTER);
        lblWeigth2.setFont(new Font("Dialog", Font.BOLD, 20));
        add(lblWeigth2);
        // Eventos de teclado para los pesos
        txtWeight0.addKeyListener(new CustomKeyListener(txtWeight0));
        txtWeight1.addKeyListener(new CustomKeyListener(txtWeight1));
        txtWeight2.addKeyListener(new CustomKeyListener(txtWeight2));
        // Eventos de movimiento de caret para los pesos
        txtWeight0.addCaretListener(new CustomCaretListener(txtWeight0, map, 0));
        txtWeight1.addCaretListener(new CustomCaretListener(txtWeight1, map, 1));
        txtWeight2.addCaretListener(new CustomCaretListener(txtWeight2, map, 2));
        // Eventos para el enfoque de los pesos
        txtWeight0.addFocusListener(new CustomFocusListener(txtWeight0, 0));
        txtWeight1.addFocusListener(new CustomFocusListener(txtWeight1, 1));
        txtWeight2.addFocusListener(new CustomFocusListener(txtWeight2, 2));
        // Boton de pesos aleatorios
        btnRandomWeights = new JButton("Pesos aleatorios");
        btnRandomWeights.setLocation(lblSubtitle.getX() + (lblSubtitle.getWidth()) - lblWeigth0.getWidth(), txtWeight0.getY());
        btnRandomWeights.setSize(txtWeight0.getWidth(), 40);
        btnRandomWeights.setBackground(new Color(71, 138, 201));
        btnRandomWeights.setOpaque(true);
        btnRandomWeights.setForeground(Color.WHITE);
        btnRandomWeights.setFont(new Font("Dialog", Font.BOLD, 12));
        btnRandomWeights.setCursor(new Cursor(Cursor.HAND_CURSOR));
        add(btnRandomWeights);
        btnRandomWeights.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                weights[0] = getRandom();
                weights[1] = getRandom();
                weights[2] = getRandom();
                txtWeight0.setText(String.valueOf(weights[0]));
                txtWeight1.setText(String.valueOf(weights[1]));
                txtWeight2.setText(String.valueOf(weights[2]));
                System.arraycopy(weights, 0, perceptronWeights, 0, weights.length);
                System.arraycopy(weights, 0, adalineWeights, 0, weights.length);
                map.repaint();
            }
        });
        // Boton para empezar el algoritmo del perceptron
        btnPerceptron = new JButton("Perceptron");
        btnPerceptron.setSize((lblSubtitle.getWidth() / 2),50);
        btnPerceptron.setLocation(lblPASubtitle.getX(), btnRandomWeights.getY() + btnRandomWeights.getHeight() + 10);
        btnPerceptron.setCursor(new Cursor(Cursor.HAND_CURSOR));
        btnPerceptron.setEnabled(false);
        btnPerceptron.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                double learningRate;
                int epochs;
                try {
                    learningRate = Double.parseDouble(txtLearningRate.getText());
                    epochs = Integer.parseInt(txtEpochs.getText());
                } catch (NumberFormatException ex) {
                    JOptionPane.showMessageDialog(null, "Parametros del perceptron no especificados o incorrectos", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                if ( epochs <= 0 ) {
                    JOptionPane.showMessageDialog(null, "Las epocas no pueden ser 0 o menos", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                if ( learningRate == 0 ) {
                    JOptionPane.showMessageDialog(null, "El factor de aprendizaje no puede ser 0", "Error", JOptionPane.ERROR_MESSAGE);
                }
                if ( points.size() < 1 ) {
                    JOptionPane.showMessageDialog(null, "Ingrese minimo una instancia", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                // Deshabilitamos la interfaz temporalmente y otras cosas
                changeUIForPerceptron(false);
                clickEnable = false;
                // Eliminamos los puntos de barrido
                ArrayList<Point> newPoints = new ArrayList<>();
                for ( Point tmpPoint : points ) {
                    if ( !tmpPoint.sweep ) {
                        newPoints.add(tmpPoint);
                    }
                }
                points = newPoints;
                map.repaint();
                // Creacion del conjunto de datos
                String[] headers = { "x_1", "x_2", "y" };
                String[] attributeTypes = { DataSet.NUMERIC_TYPE, DataSet.NUMERIC_TYPE, DataSet.NUMERIC_TYPE };
                DataSet dataSet;
                try {
                    dataSet = DataSet.getEmptyDataSetWithHeaders(headers, attributeTypes, "y");
                } catch (Exception ex) {
                    System.out.println("El dataset no pudo ser creado");
                    return;
                }
                for ( Point point : points ) {
                    try {
                        dataSet.addInstance(new ArrayList<>(Arrays.asList("" + point.x,"" + point.y, "" + ((point.leftClick) ? 0 : 1))));
                    } catch (Exception ex) {
                        System.out.println("No se pudo agregar la instancia del punto " + point);
                    }
                }
                System.out.println("Conjunto de datos con el que el algoritmo trabajara");
                System.out.println(dataSet);
                // Generamos los parametros y modelo del perceptron
                PerceptronThread.Params params = new PerceptronThread.Params();
                params.setEpochs(epochs);
                params.setLearningRate(learningRate);
                params.setWeights(perceptronWeights);
                try {
                    PerceptronThread perceptronThread = new PerceptronThread();
                    perceptronThread.makeModel(dataSet, params, AdalineWindow.this);
                } catch (Exception ex) {
                    System.out.println("El modelo no se pudo generar");
                }
            }
        });
        add(btnPerceptron);
        // Boton para empezar el algoritmo del adaline
        btnAdaline = new JButton("Adaline");
        btnAdaline.setSize(btnPerceptron.getSize());
        btnAdaline.setLocation(btnPerceptron.getX() + btnPerceptron.getWidth(), btnPerceptron.getY());
        btnAdaline.setCursor(new Cursor(Cursor.HAND_CURSOR));
        btnAdaline.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                double learningRate;
                int epochs;
                double minError;
                try {
                    learningRate = Double.parseDouble(txtLearningRate.getText());
                    epochs = Integer.parseInt(txtEpochs.getText());
                    minError = Double.parseDouble(txtMinError.getText());
                } catch (NumberFormatException ex) {
                    JOptionPane.showMessageDialog(null, "Parametros del adaline no especificados o incorrectos", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                if ( epochs <= 0 ) {
                    JOptionPane.showMessageDialog(null, "Las epocas no pueden ser 0 o menos", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                if ( learningRate == 0 ) {
                    JOptionPane.showMessageDialog(null, "El factor de aprendizaje no puede ser 0", "Error", JOptionPane.ERROR_MESSAGE);
                }
                if ( points.size() < 1 ) {
                    JOptionPane.showMessageDialog(null, "Ingrese minimo una instancia", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                // Deshabilitamos la interfaz temporalmente y otras cosas
                changeUIForPerceptron(false);
                clickEnable = false;
                // Eliminamos los puntos de barrido
                ArrayList<Point> newPoints = new ArrayList<>();
                for ( Point tmpPoint : points ) {
                    if ( !tmpPoint.sweep ) {
                        newPoints.add(tmpPoint);
                    }
                }
                points = newPoints;
                map.repaint();
                // Creacion del conjunto de datos
                String[] headers = { "x_1", "x_2", "y" };
                String[] attributeTypes = { DataSet.NUMERIC_TYPE, DataSet.NUMERIC_TYPE, DataSet.NUMERIC_TYPE };
                DataSet dataSet;
                try {
                    dataSet = DataSet.getEmptyDataSetWithHeaders(headers, attributeTypes, "y");
                } catch (Exception ex) {
                    System.out.println("El dataset no pudo ser creado");
                    return;
                }
                for ( Point point : points ) {
                    try {
                        dataSet.addInstance(new ArrayList<>(Arrays.asList("" + point.x,"" + point.y, "" + ((point.leftClick) ? 0 : 1))));
                    } catch (Exception ex) {
                        System.out.println("No se pudo agregar la instancia del punto " + point);
                    }
                }
                System.out.println("Conjunto de datos con el que el algoritmo trabajara");
                System.out.println(dataSet);
                AdalineThread.Params params = new AdalineThread.Params();
                params.setEpochs(epochs);
                params.setMinError(minError);
                params.setLearningRate(learningRate);
                params.setWeights(adalineWeights);
                params.setFunction(Adaline.SIGMOID_FUNCTION);
                try {
                    AdalineThread adalineThread = new AdalineThread();
                    adalineThread.makeModel(dataSet, params, AdalineWindow.this);
                    if ( errorChartAdalineWindow != null )  {
                        errorChartAdalineWindow.dispose();
                    }
                    errorChartAdalineWindow = new ErrorChartAdalineWindow(AdalineWindow.this);
                } catch (Exception ex) {
                    System.out.println("El modelo no se pudo generar");
                }
            }
        });
        add(btnAdaline);
        /** Resultados */
        // Separador
        JSeparator jsResults = new JSeparator();
        jsResults.setOrientation(SwingConstants.HORIZONTAL);
        jsResults.setSize(lblSubtitle.getWidth(), 2);
        jsResults.setLocation(lblSubtitle.getX(), btnAdaline.getY() + btnAdaline.getHeight() + 10);
        add(jsResults);
        // Titulo de los resultados
        JLabel lblResults = new JLabel("Resultados");
        lblResults.setLocation(lblSubtitle.getX(), jsResults.getY() + jsResults.getHeight() + 10);
        lblResults.setSize(lblSubtitle.getWidth(), 18);
        lblResults.setHorizontalAlignment(JLabel.CENTER);
        lblResults.setFont(new Font("Dialog", Font.BOLD, 14));
        add(lblResults);
        // Titulo para el perceptron
        JLabel lblPerceptronResults = new JLabel("Perceptron");
        lblPerceptronResults.setLocation(lblPASubtitle.getX(), lblResults.getY() + lblResults.getHeight() + 10);
        lblPerceptronResults.setSize(lblPASubtitle.getWidth(), 18);
        lblPerceptronResults.setHorizontalAlignment(JLabel.CENTER);
        lblPerceptronResults.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblPerceptronResults);
        // Titulo para el adaline
        JLabel lblAdalineResults = new JLabel("Adaline");
        lblAdalineResults.setLocation(lblAdalineSubtitule.getX(), lblPerceptronResults.getY());
        lblAdalineResults.setSize(lblPerceptronResults.getWidth(), 18);
        lblAdalineResults.setHorizontalAlignment(JLabel.CENTER);
        lblAdalineResults.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblAdalineResults);
        // Perceptron - Epoca
        lblPEpochResult = new JLabel("<html>Epoca: <b>0</b></html>");
        lblPEpochResult.setLocation(lblPerceptronResults.getX(), lblPerceptronResults.getY() + lblPerceptronResults.getHeight() + 5);
        lblPEpochResult.setSize(lblPerceptronResults.getWidth(), 18);
        lblPEpochResult.setHorizontalAlignment(JLabel.LEFT);
        lblPEpochResult.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblPEpochResult);
        // Perceptron - Peso 0
        lblPWeightResult0 = new JLabel("<html>w" + unicodeSubscript0 + " = <b>0.0</b></html>");
        lblPWeightResult0.setLocation(lblPEpochResult.getX(), lblPEpochResult.getY() + lblPEpochResult.getHeight() + 5);
        lblPWeightResult0.setSize(lblPerceptronResults.getWidth(), 18);
        lblPWeightResult0.setHorizontalAlignment(JLabel.LEFT);
        lblPWeightResult0.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblPWeightResult0);
        // Perceptron - Peso 1
        lblPWeightResult1 = new JLabel("<html>w" + unicodeSubscript1 + " = <b>0.0</b></html>");
        lblPWeightResult1.setLocation(lblPEpochResult.getX(), lblPWeightResult0.getY() + lblPWeightResult0.getHeight() + 5);
        lblPWeightResult1.setSize(lblPerceptronResults.getWidth(), 18);
        lblPWeightResult1.setHorizontalAlignment(JLabel.LEFT);
        lblPWeightResult1.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblPWeightResult1);
        // Perceptron - Peso 2
        lblPWeightResult2 = new JLabel("<html>w" + unicodeSubscript2 + " = <b>0.0</b></html>");
        lblPWeightResult2.setLocation(lblPEpochResult.getX(), lblPWeightResult1.getY() + lblPWeightResult1.getHeight() + 5);
        lblPWeightResult2.setSize(lblPerceptronResults.getWidth(), 18);
        lblPWeightResult2.setHorizontalAlignment(JLabel.LEFT);
        lblPWeightResult2.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblPWeightResult2);
        // Adaline - Epoca
        lblAEpochResult = new JLabel("<html>Epoca: <b>0</b></html>");
        lblAEpochResult.setLocation(lblAdalineResults.getX(), lblPerceptronResults.getY() + lblPerceptronResults.getHeight() + 5);
        lblAEpochResult.setSize(lblAdalineResults.getWidth(), 18);
        lblAEpochResult.setHorizontalAlignment(JLabel.LEFT);
        lblAEpochResult.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblAEpochResult);
        // Adaline - Peso 0
        lblAWeightResult0 = new JLabel("<html>w" + unicodeSubscript0 + " = <b>0.0</b></html>");
        lblAWeightResult0.setLocation(lblAEpochResult.getX(), lblAEpochResult.getY() + lblAEpochResult.getHeight() + 5);
        lblAWeightResult0.setSize(lblAdalineResults.getWidth(), 18);
        lblAWeightResult0.setHorizontalAlignment(JLabel.LEFT);
        lblAWeightResult0.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblAWeightResult0);
        // Adaline - Peso 1
        lblAWeightResult1 = new JLabel("<html>w" + unicodeSubscript1 + " = <b>0.0</b></html>");
        lblAWeightResult1.setLocation(lblAEpochResult.getX(), lblAWeightResult0.getY() + lblAWeightResult0.getHeight() + 5);
        lblAWeightResult1.setSize(lblAdalineResults.getWidth(), 18);
        lblAWeightResult1.setHorizontalAlignment(JLabel.LEFT);
        lblAWeightResult1.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblAWeightResult1);
        // Adaline - Peso 2
        lblAWeightResult2 = new JLabel("<html>w" + unicodeSubscript2 + " = <b>0.0</b></html>");
        lblAWeightResult2.setLocation(lblAEpochResult.getX(), lblAWeightResult1.getY() + lblAWeightResult1.getHeight() + 5);
        lblAWeightResult2.setSize(lblAdalineResults.getWidth(), 18);
        lblAWeightResult2.setHorizontalAlignment(JLabel.LEFT);
        lblAWeightResult2.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblAWeightResult2);
        // Adaline - Error
        lblAErrorResult = new JLabel("<html>Error: <b>0.0</b></html>");
        lblAErrorResult.setLocation(lblAdalineResults.getX(), lblAWeightResult2.getY() + lblAWeightResult2.getHeight() + 5);
        lblAErrorResult.setSize(lblAdalineResults.getWidth(), 18);
        lblAErrorResult.setHorizontalAlignment(JLabel.LEFT);
        lblAErrorResult.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblAErrorResult);

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setResizable(false);
        setVisible(true);
    }

    private void changeUIForPerceptron(boolean enable) {
        btnPerceptron.setEnabled(enable);
        btnAdaline.setEnabled(enable);
        txtWeight0.setEditable(enable);
        txtWeight1.setEditable(enable);
        txtWeight2.setEditable(enable);
        txtLearningRate.setEditable(enable);
        txtEpochs.setEditable(enable);
        txtMinError.setEditable(enable);
        btnRandomWeights.setEnabled(enable);
        jmOptions.setEnabled(enable);
        jmPredict.setEnabled(enable);
    }

    private double getRandom() {
        int random = (int) (Math.random() * (100 * (int) 1.0));
        int sign = (int) (Math.random() * 10);
        return (double) ((sign % 2 == 0) ? -random : random) / 100;
    }

    public void updateEpochForPerceptron(int epoch, boolean done, boolean stop) {
        if ( stop ) {
            if ( done ) {
                lblPEpochResult.setText("<html>Convergio en la epoca: <b>" + epoch + "</b></html>");
            } else {
                lblPEpochResult.setText("<html>No convergio. Epocas: <b>" + epoch + "</b></html>");
            }
        } else {
            lblPEpochResult.setText("<html>Epoca: <b>" + epoch + "</b></html>");
        }
    }

    public void updateWeightsForPerceptron() {
        lblPWeightResult0.setText("<html>w" + unicodeSubscript0 + " = <b>" + perceptronWeights[0] + "</b></html>");
        lblPWeightResult1.setText("<html>w" + unicodeSubscript1 + " = <b>" + perceptronWeights[1] + "</b></html>");
        lblPWeightResult2.setText("<html>w" + unicodeSubscript2 + " = <b>" + perceptronWeights[2] + "</b></html>");
        map.repaint();
    }

    public void setModelForPerceptron(PerceptronThread.Model model) {
        pModel = model;
        System.out.println("Modelo obtenido");
        System.out.println(model);
        changeUIForPerceptron(true);
        clickEnable = true;
        changeWeights = false;
        perceptronModelEnable = true;
        addInstanceEnable = false;
        jmPredict.setVisible(true);
        jmPerceptron.setVisible(true);
        jmiPerceptronPredict.setSelected(true);
        selectedPerceptron = true;
        selectedAdaline = false;
    }

    public void updateEpochForAdaline(int epoch, boolean done, boolean stop) {
        if ( stop ) {
            if ( done ) {
                lblAEpochResult.setText("<html>Convergio en la epoca: <b>" + epoch + "</b></html>");
            } else {
                lblAEpochResult.setText("<html>No convergio. Epocas: <b>" + epoch + "</b></html>");
            }
        } else {
            lblAEpochResult.setText("<html>Epoca: <b>" + epoch + "</b></html>");
        }
    }

    public void addErrorForChart(int epoch, double error) {
        if ( errorChartAdalineWindow != null ) {
            errorChartAdalineWindow.addValueForSeries(epoch, error);
        }
    }

    public void updateErrorForAdaline(double error) {
        lblAErrorResult.setText("<html>Error: <b>" + error + "</b></html>");
    }

    public void updateWeightsForAdaline() {
        lblAWeightResult0.setText("<html>w" + unicodeSubscript0 + " = <b>" + adalineWeights[0] + "</b></html>");
        lblAWeightResult1.setText("<html>w" + unicodeSubscript1 + " = <b>" + adalineWeights[1] + "</b></html>");
        lblAWeightResult2.setText("<html>w" + unicodeSubscript2 + " = <b>" + adalineWeights[2] + "</b></html>");
        map.repaint();
    }

    public void setModelForAdaline(AdalineThread.Model model) {
        aModel = model;
        System.out.println("Modelo obtenido");
        System.out.println(model);
        changeUIForPerceptron(true);
        clickEnable = true;
        changeWeights = false;
        adalineModelEnable = true;
        btnPerceptron.setEnabled(true);
        addInstanceEnable = false;
        jmPredict.setVisible(true);
        jmAdaline.setVisible(true);
        jmiAdalinePredict.setSelected(true);
        selectedAdaline = true;
        selectedPerceptron = false;
        showConfusionMatrix();
    }

    private void showConfusionMatrix() {
        int tp = 0;
        int tn = 0;
        int fp = 0;
        int fn = 0;
        for ( Point point : points ) {
            if ( point.sweep ) {
                continue;
            }
            double result = -1;
            try {
                result = aModel.predict(new Object[]{ point.x, point.y });
            } catch (Exception e) {
                System.out.println("No se pudo realizar la matriz de confusion");
                return;
            }
            int predict = result > 0.5 ? 1 : 0;
            if ( point.leftClick ) {
                if ( predict == 0 ) {
                    tn++;
                } else {
                    fp++;
                }
            } else {
                if ( predict == 1 ) {
                    tp++;
                } else {
                    fn++;
                }
            }
        }
        System.out.println();
        System.out.println("Matriz de confusion");
        System.out.println(" _______________________________________________________________");
        System.out.println("|               |   POSITIVO    |   NEGATIVO    |     TOTAL     |");
        System.out.println("|_______________|_______________|_______________|_______________|");
        System.out.println("|               |      TP       |      FP       |               |");
        System.out.println("|   POSITIVO    | "+String.format("% 13d",tp)+" | "+String.format("% 13d", fp)+" | "+String.format("% 13d", tp+fp)+" |");
        System.out.println("|_______________|_______________|_______________|_______________|");
        System.out.println("|               |      FN       |      TN       |               |");
        System.out.println("|   NEGATIVO    | "+String.format("% 13d", fn)+" | "+String.format("% 13d", tn)+" | "+String.format("% 13d", fn + tn)+" |");
        System.out.println("|_______________|_______________|_______________|_______________|");
        System.out.println("|               |               |               |               |");
        System.out.println("|     TOTAL     | "+String.format("% 13d", tp + fn)+" | "+String.format("% 13d", fp + tn)+" | "+String.format("% 13d", fn + tn + tp + fp)+" |");
        System.out.println("|_______________|_______________|_______________|_______________|");
    }

    private class Map extends JPanel {

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            // Obtenemos el alto y ancho del componente
            int width = getWidth();
            int height = getHeight();
            // Linea vertical del lienzo
            g.drawLine(width / 2, 0, width / 2, height);
            // Linea horizontal
            g.drawLine(0, height / 2, width, height / 2);
            // ALgunas líneas más de apoyo
            g.setColor(new Color(170, 183, 184));
            for ( int i = 1; i < (width / MAP_SCALE / 10); i++ ) {
                int point = (int) (i * width / MAP_SCALE / 2);
                if ( point == width / 2 )  {
                    continue;
                }
                g.drawLine(point, 0, point, height);
                g.drawLine(0, point, width, point);
            }
            // Dibujamos los puntos hasta ahora obtenidos
            for ( Point point : points ) {
                if ( point.leftClick ) {
                    if ( point.sweep ) {
                        g.setColor(point.color);
                        if ( point.adaline ) {
                            g.fillRect(point.xMap, point.yMap, 1, 1);
                        } else {
                            g.drawOval(point.xMap - RADIUS_POINT, point.yMap - RADIUS_POINT, RADIUS_POINT * 2, RADIUS_POINT * 2);
                        }
                    } else {
                        g.setColor(Color.BLACK);
                        g.fillOval((point.xMap - RADIUS_POINT) - 2, (point.yMap - RADIUS_POINT) - 2, (RADIUS_POINT * 2) + 4, (RADIUS_POINT * 2) + 4);
                        g.setColor(point.color);
                        g.fillOval(point.xMap - RADIUS_POINT, point.yMap - RADIUS_POINT, RADIUS_POINT * 2, RADIUS_POINT * 2);
                    }
                } else {
                    if ( point.sweep ) {
                        g.setColor(point.color);
                        if ( point.adaline ) {
                            g.fillRect(point.xMap, point.yMap, 1, 1);
                        } else {
                            g.drawRect(point.xMap - RADIUS_POINT, point.yMap - RADIUS_POINT, RADIUS_POINT * 2, RADIUS_POINT * 2);
                        }
                    } else {
                        g.setColor(Color.BLACK);
                        g.fillRect((point.xMap - RADIUS_POINT) - 1, (point.yMap - RADIUS_POINT) - 1, (RADIUS_POINT * 2) + 2, (RADIUS_POINT * 2) + 2);
                        g.setColor(point.color);
                        g.fillRect(point.xMap - RADIUS_POINT, point.yMap - RADIUS_POINT, RADIUS_POINT * 2, RADIUS_POINT * 2);
                    }
                }
            }
            // Dibujamos la línea del perceptron
            g.setColor(Color.RED);
            drawLineInPlane(perceptronWeights, g);
            // Dibujamos la linea del adaline
            g.setColor(Color.MAGENTA);
            drawLineInPlane(adalineWeights, g);
        }

    }

    private void drawLineInPlane(double[] weights, Graphics g) {
        double x1_1 = MAP_SCALE + 1.0;
        double x2_1 = Double.NaN;
        double x1_2 = -MAP_SCALE - 1.0;
        double x2_2 = Double.NaN;
        if ( weights[1] == 0.0 && weights[2] != 0.0 || weights[1] != 0.0 && weights[2] != 0.0 ) {
            x2_1 = ( -weights[1] * x1_1 + weights[0] ) / weights[2];
            x2_2 = ( -weights[1] * x1_2 + weights[0] ) / weights[2];
        }
        if ( weights[1] != 0.0 && weights[2] == 0.0 ) {
            x2_1 = ( -weights[2] * x1_1 + weights[0] ) / weights[1];
            x2_2 = ( -weights[2] * x1_2 + weights[0] ) / weights[1];
        }
        // Transformamos el punto a las coordenadas del mapa
        x1_1 = ( x1_1 * (MAP_WIDTH * 0.5) / MAP_SCALE ) + (MAP_WIDTH * 0.5);
        x2_1 *= (MAP_HEIGHT * 0.5) / MAP_SCALE;
        x2_1 = ( x2_1 > 0 ) ? (MAP_HEIGHT * 0.5) - x2_1 : (MAP_HEIGHT * 0.5) + Math.abs(x2_1);
        x1_2 = ( x1_2 * (MAP_WIDTH * 0.5) / MAP_SCALE ) + (MAP_WIDTH * 0.5);
        x2_2 *= (MAP_HEIGHT * 0.5) / MAP_SCALE;
        x2_2 = ( x2_2 > 0 ) ? (MAP_HEIGHT * 0.5) - x2_2 : (MAP_HEIGHT * 0.5) + Math.abs(x2_2);
        Graphics2D graphics2D = (Graphics2D) g;
        graphics2D.setStroke(new BasicStroke(2));
        graphics2D.draw(new Line2D.Double(x1_1,x2_1,x1_2,x2_2));
    }

    private static class Point {

        public int xMap;
        public int yMap;
        public double x;
        public double y;
        public boolean leftClick;
        public boolean sweep;
        public Color color;
        public boolean adaline;

        @Override
        public String toString() {
            return "Point{" +
                    "xMap=" + xMap +
                    ", yMap=" + yMap +
                    ", x=" + x +
                    ", y=" + y +
                    ", leftClick=" + leftClick +
                    ", sweep=" + sweep +
                    ", color=" + color +
                    ", adaline=" + adaline +
                    '}';
        }

    }

    private static class CustomKeyListener extends KeyAdapter {

        private final JTextField txtField;

        public CustomKeyListener(JTextField txtField)
        {
            this.txtField = txtField;
        }

        @Override
        public void keyTyped(KeyEvent e) {
            if ( (e.getKeyChar() < '0' || e.getKeyChar() > '9') && ( e.getKeyChar() != '.' && e.getKeyChar() != '-' ) ) {
                e.consume();
            }
            if ( e.getKeyChar() == '-' && ( txtField.getCaretPosition() != 0 || txtField.getText().contains("-") ) ) {
                e.consume();
            }
            if ( e.getKeyChar() == '.' && !txtField.getText().isEmpty() && ( txtField.getText().contains(".") ) ) {
                e.consume();
            }
            if ( txtField.getText().startsWith("-") && txtField.getCaretPosition() == 0 ) {
                e.consume();
            }
            super.keyTyped(e);
        }
    }

    private class CustomCaretListener implements CaretListener {

        private final JTextField txtField;
        private final JPanel map;
        private final int idxWeight;

        public CustomCaretListener(JTextField txtField, JPanel map, int idxWeight)
        {
            this.txtField = txtField;
            this.map = map;
            this.idxWeight = idxWeight;
        }

        @Override
        public void caretUpdate(CaretEvent e) {
            try {
                double weigth = Double.parseDouble(txtField.getText());
                if ( weigth != weights[idxWeight] ) {
                    weights[idxWeight] = weigth;
                    changeWeights = true;
                    System.arraycopy(weights, 0, perceptronWeights, 0, weights.length);
                    System.arraycopy(weights, 0, adalineWeights, 0, weights.length);
                }
            } catch (Exception ex) {
                weights[idxWeight] = 0.0;
                changeWeights = true;
                System.arraycopy(weights, 0, perceptronWeights, 0, weights.length);
                System.arraycopy(weights, 0, adalineWeights, 0, weights.length);
            } finally {
                if ( (adalineModelEnable || perceptronModelEnable) && changeWeights ) {
                    adalineModelEnable = false;
                    perceptronModelEnable = false;
                    jmPredict.setVisible(false);
                    addInstanceEnable = true;
                    btnPerceptron.setEnabled(false);
                    jmAdaline.setVisible(false);
                    jmPerceptron.setVisible(false);
                    ArrayList<Point> newPoints = new ArrayList<>();
                    for ( Point tmpPoint : points ) {
                        if ( !tmpPoint.sweep ) {
                            newPoints.add(tmpPoint);
                        }
                    }
                    points = newPoints;
                }
                map.repaint();
            }
        }

    }

    private class CustomFocusListener implements FocusListener {

        private final JTextField txtField;
        private final int idxWeight;

        public CustomFocusListener(JTextField txtField, int idxWeight)
        {
            this.txtField = txtField;
            this.idxWeight = idxWeight;
        }

        @Override
        public void focusGained(FocusEvent e) {

        }

        @Override
        public void focusLost(FocusEvent e) {
            if ( txtField.getText().isEmpty() ) {
                txtField.setText(String.valueOf(weights[idxWeight]));
            }
        }
    }

}
