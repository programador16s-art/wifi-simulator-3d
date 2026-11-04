CREATE TABLE IF NOT EXISTS materiales (
    id_material VARCHAR(40) PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    atenuacion_2_4ghz_db DECIMAL(7, 2) NOT NULL,
    atenuacion_5ghz_db DECIMAL(7, 2) NOT NULL,
    descripcion TEXT NOT NULL,
    procedencia VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS simulaciones (
    id_simulacion CHAR(36) PRIMARY KEY,
    fecha TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    frecuencia_ghz DECIMAL(5, 2) NOT NULL,
    potencia_transmision_dbm DECIMAL(7, 2) NOT NULL,
    ancho_banda_hz BIGINT NOT NULL,
    dimensiones_x_m DECIMAL(7, 2) NOT NULL,
    dimensiones_y_m DECIMAL(7, 2) NOT NULL,
    dimensiones_z_m DECIMAL(7, 2) NOT NULL,
    resolucion_espacial_m DECIMAL(5, 2) NOT NULL
);

CREATE TABLE IF NOT EXISTS configuraciones (
    id_configuracion BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    id_simulacion CHAR(36) NOT NULL,
    router_x_m DECIMAL(7, 2) NOT NULL,
    router_y_m DECIMAL(7, 2) NOT NULL,
    router_z_m DECIMAL(7, 2) NOT NULL,
    parametros_json JSON NOT NULL,
    CONSTRAINT fk_configuracion_simulacion
        FOREIGN KEY (id_simulacion) REFERENCES simulaciones(id_simulacion)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS obstaculos (
    id_registro BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    id_simulacion CHAR(36) NOT NULL,
    id_obstaculo VARCHAR(80) NOT NULL,
    tipo VARCHAR(40) NOT NULL,
    material VARCHAR(40) NOT NULL,
    posicion_x_m DECIMAL(7, 2) NOT NULL,
    posicion_y_m DECIMAL(7, 2) NOT NULL,
    posicion_z_m DECIMAL(7, 2) NOT NULL,
    dimensiones_json JSON NOT NULL,
    parametros_json JSON NOT NULL,
    CONSTRAINT fk_obstaculo_simulacion
        FOREIGN KEY (id_simulacion) REFERENCES simulaciones(id_simulacion)
        ON DELETE CASCADE,
    CONSTRAINT uq_obstaculo_simulacion UNIQUE (id_simulacion, id_obstaculo)
);

INSERT INTO materiales
    (id_material, nombre, atenuacion_2_4ghz_db, atenuacion_5ghz_db, descripcion, procedencia)
VALUES
    ('concrete', 'Concreto', 12.00, 18.00, 'Muro sólido de concreto.', 'Estimación orientativa configurable; no es una medición experimental.'),
    ('reinforced_concrete', 'Concreto armado', 18.00, 25.00, 'Concreto con refuerzo metálico.', 'Estimación orientativa configurable; no es una medición experimental.'),
    ('platabanda', 'Platabanda', 15.00, 22.00, 'Losa o cubierta estructural; valor orientativo.', 'Estimación orientativa configurable; no es una medición experimental.'),
    ('zinc', 'Lámina de zinc', 20.00, 28.00, 'Lámina metálica delgada.', 'Estimación orientativa configurable; no es una medición experimental.'),
    ('glass', 'Vidrio', 3.00, 5.00, 'Vidrio común; no modela recubrimientos.', 'Estimación orientativa configurable; no es una medición experimental.'),
    ('wood', 'Madera', 4.00, 6.00, 'Panel o puerta de madera.', 'Estimación orientativa configurable; no es una medición experimental.'),
    ('drywall', 'Drywall / yeso', 3.00, 5.00, 'Tabique liviano.', 'Estimación orientativa configurable; no es una medición experimental.'),
    ('brick', 'Ladrillo', 8.00, 12.00, 'Pared de ladrillo.', 'Estimación orientativa configurable; no es una medición experimental.'),
    ('metal', 'Metal', 20.00, 28.00, 'Obstáculo metálico.', 'Estimación orientativa configurable; no es una medición experimental.')
ON DUPLICATE KEY UPDATE
    nombre = VALUES(nombre),
    atenuacion_2_4ghz_db = VALUES(atenuacion_2_4ghz_db),
    atenuacion_5ghz_db = VALUES(atenuacion_5ghz_db),
    descripcion = VALUES(descripcion),
    procedencia = VALUES(procedencia);
