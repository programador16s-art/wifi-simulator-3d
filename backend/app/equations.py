"""Symbolic reference equations used to document and verify the numeric model."""

import sympy as sp

distance_km, frequency_mhz = sp.symbols("distance_km frequency_mhz", positive=True)
fspl_db_expression = (
    20 * sp.log(distance_km, 10)
    + 20 * sp.log(frequency_mhz, 10)
    + sp.Float("32.44")
)

tx_dbm, tx_gain_db, tx_loss_db, rx_gain_db, obstruction_loss_db = sp.symbols(
    "tx_dbm tx_gain_db tx_loss_db rx_gain_db obstruction_loss_db", real=True
)
rssi_dbm_expression = (
    tx_dbm + tx_gain_db - tx_loss_db - fspl_db_expression
    - obstruction_loss_db + rx_gain_db
)
