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
print(a, b)

# Calcular media y varianza normalizada
mu_norm = a / (a + b)
var_norm = (a * b) / ((a + b) ** 2 * (a + b + 1))
std_norm = np.sqrt(var_norm)
print(mu_norm, std_norm)

# Calcular media y desviación estándar sin normalizar
mu = (max_sales - min_sales) * mu_norm + min_sales
var = (max_sales - min_sales) ** 2 * var_norm
sigma = np.sqrt(var)

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
# Se ajusta para uso comercial, tarifa de consumo excedente
# Consumo básico: 0.767 pesos/kWh, Consumo excedente: 3.672 pesos/kWh
consumo_kw_mes = 120000  # Estimación de 120,000 kWh mensuales
gasto_luz = (consumo_kw_mes * 0.767) + (consumo_kw_mes * 3.672)
print("Gasto de luz:", gasto_luz)

# Nuevos gastos agregados
# Gasto en mantenimiento de equipos
gasto_mantenimiento_equipos = 50000  # Costo mensual de mantenimiento de refrigeradores, AC, etc.

# Gasto en insumos y suministros generales
gasto_insumos = 70000  # Costo mensual de insumos como productos de limpieza, papelería, etc.

# Calcular gastos totales incluyendo los nuevos gastos
gastos_tot = gasto_luz + nomina_total + gasto_mantenimiento_equipos + gasto_insumos
print("Gastos totales:", gastos_tot)

# Distribución normal
omega = norm.ppf(0.01)

# Ingreso esperado
ingreso = gastos_tot + 2500000  

a_ = mu ** 2
b_ = -2 * mu * ingreso - omega * 2 * sigma ** 2
c_ = ingreso ** 2

discriminante = b_ ** 2 - 4 * a_ * c_

if discriminante >= 0:
    N1 = (-b_ + np.sqrt(discriminante)) / (2 * a_)
    N2 = (-b_ - np.sqrt(discriminante)) / (2 * a_)
    print("N1:", N1)
    print("N2:", N2)
else:
    print("El discriminante es negativo. No se puede calcular la raíz cuadrada.")
    print("Discriminante:", discriminante)

# ----------- Análisis para las preguntas 7 y 8 -----------------

# Supongamos que el rating promedio de la comunidad es de 8.2, con desviación estándar 0.5
mu_rating = 8.2  # Media de los ratings en la comunidad
sigma_rating = 0.5  # Desviación estándar de los ratings
n_sucursal = 50  # Tamaño de la muestra en la nueva sucursal

# Queremos saber la probabilidad de que el rating promedio sea 8.5 o más
rating_objetivo = 8.5

# Aplicar el Teorema del Límite Central para calcular la probabilidad
sigma_muestra = sigma_rating / np.sqrt(n_sucursal)
z = (rating_objetivo - mu_rating) / sigma_muestra

# Probabilidad de que el rating promedio sea mayor o igual a 8.5
probabilidad_rating = 1 - norm.cdf(z)
print(f'Valor z: {z}')
print(f'Probabilidad de que el rating promedio sea 8.5 o más: {probabilidad_rating:.4f}')


