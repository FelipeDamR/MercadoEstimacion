# Código completo con todos los cálculos y resultados impresos

import pandas as pd
import numpy as np
from scipy.stats import beta, norm

df = pd.read_csv(r'C:\Users\fdx11\Desktop\VsCode_Proyectos\python\SuperMarketData.csv')


# Convertir de dólares a pesos
sales = np.array(df["Sales"]) * 19.88

# Calcular ventas máximas y mínimas
max_sales = np.max(sales)
min_sales = np.min(sales)

# Normalización de ventas
sales_norm = (1 / (max_sales - min_sales)) * sales

# Ajuste de la distribución beta a los datos de ventas (sin normalizar)
a, b, loc, scale = beta.fit(sales)
print(f"Ajuste Beta: a = {a:.4f}, b = {b:.4f}")

# Calcular media y varianza normalizada
mu_norm = a / (a + b)
var_norm = (a * b) / ((a + b) ** 2 * (a + b + 1))
std_norm = np.sqrt(var_norm)
print(f"Media normalizada: {mu_norm:.4f}, Desviación estándar normalizada: {std_norm:.4f}")

# Calcular media y desviación estándar sin normalizar
mu = (max_sales - min_sales) * mu_norm + min_sales
var = (max_sales - min_sales) ** 2 * var_norm
sigma = np.sqrt(var)
print(f"Media sin normalizar: {mu:.2f} pesos, Desviación estándar sin normalizar: {sigma:.2f} pesos")

# Salarios de los trabajadores
dias_trab = 24  # Días trabajados
fact = 1.15

sal_cajeros = 258.25  # Salario por día de los cajeros
num_cajeros = 35  # Ajuste del número de cajeros
tot_sal_cajeros = sal_cajeros * num_cajeros * dias_trab * fact

sal_conserjes = 5000  # Salario mensual de los conserjes
num_conserjes = 25  # Ajuste del número de conserjes
tot_sal_conserjes = sal_conserjes * num_conserjes * fact

tot_sal_gerente = 100000  # Salario mensual del gerente

sub_gerente = 45000  # Salario mensual de subgerentes
num_sub_gerente = 5  # Ajuste del número de subgerentes
tot_sal_sub_gerente = sub_gerente * num_sub_gerente

sal_almacenista = 262.13  # Salario diario de los almacenistas
almacenista = 45  # Ajuste del número de almacenistas
tot_sal_almacenista = sal_almacenista * almacenista * dias_trab * fact

g_pasillo = 264.65  # Salario diario de trabajadores de pasillo
num_pasillo = 45  # Ajuste del número de trabajadores de pasillo
tot_sal_pasillo = g_pasillo * num_pasillo * dias_trab * fact

# Agregamos guardias de seguridad y técnicos de mantenimiento
sal_guardia = 400  # Salario diario de los guardias de seguridad
num_guardias = 10  # Número de guardias de seguridad
tot_sal_guardias = sal_guardia * num_guardias * dias_trab * fact

sal_tecnico = 500  # Salario diario de los técnicos de mantenimiento
num_tecnicos = 8  # Número de técnicos de mantenimiento
tot_sal_tecnicos = sal_tecnico * num_tecnicos * dias_trab * fact

# Calcular nómina total incluyendo nuevos roles
nomina_total = (tot_sal_cajeros + tot_sal_conserjes + tot_sal_gerente +
                tot_sal_sub_gerente + tot_sal_almacenista + tot_sal_pasillo +
                tot_sal_guardias + tot_sal_tecnicos)
print("Nómina total:", nomina_total)

# Calcular gasto de luz ajustado a referencias más confiables de la CFE
# Consumo básico: 0.767 pesos/kWh, Consumo excedente: 3.672 pesos/kWh
consumo_kw_mes = 120000  # Estimación de 120,000 kWh mensuales
gasto_luz = (consumo_kw_mes * 0.767) + (consumo_kw_mes * 3.672)
print("Gasto de luz:", gasto_luz)

# Nuevos gastos agregados
gasto_mantenimiento_equipos = 50000  # Costo mensual de mantenimiento
gasto_insumos = 70000  # Costo mensual de insumos como productos de limpieza

# Calcular gastos totales incluyendo los nuevos gastos
gastos_tot = gasto_luz + nomina_total + gasto_mantenimiento_equipos + gasto_insumos
print("Gastos totales:", gastos_tot)

# Distribución normal para ingresos esperados
omega = norm.ppf(0.01)
ingreso = gastos_tot + 2500000  
a_ = mu ** 2
b_ = -2 * mu * ingreso - omega * 2 * sigma ** 2
c_ = ingreso ** 2
discriminante = b_ ** 2 - 4 * a_ * c_

if discriminante >= 0:
    N1 = (-b_ + np.sqrt(discriminante)) / (2 * a_)
    N2 = (-b_ - np.sqrt(discriminante)) / (2 * a_)
    print("N1:", N1, "N2:", N2)
else:
    print("El discriminante es negativo. No se puede calcular la raíz cuadrada.")
    print("Discriminante:", discriminante)

# Rating promedio en la comunidad
mu_rating = df["Rating"].mean()
sigma_rating = df["Rating"].std()
n_sucursal = 50
rating_objetivo = 8.5
sigma_muestra = sigma_rating / np.sqrt(n_sucursal)
z = (rating_objetivo - mu_rating) / sigma_muestra
probabilidad_rating = 1 - norm.cdf(z)
print(f"Probabilidad de que el rating promedio sea 8.5 o más: {probabilidad_rating:.4f}")

# Calcular el número de clientes necesarios por mes y el porcentaje de la población
gastos_totales = gastos_tot
ingreso_por_cliente = 500
frecuencia_visitas = 4
poblacion_juriquilla = 20912

clientes_necesarios_mes = gastos_totales / (ingreso_por_cliente * frecuencia_visitas)
porcentaje_poblacion = (clientes_necesarios_mes / poblacion_juriquilla) * 100

print(f"Clientes necesarios por mes: {clientes_necesarios_mes:.0f}")
print(f"Porcentaje de la población que debe ser cliente: {porcentaje_poblacion:.2f}%")
