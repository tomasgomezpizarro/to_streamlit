# Dashboard de Opciones con Streamlit

Este proyecto crea un dashboard interactivo en Streamlit para comparar métodos de diferencias finitas (Explícito e Implícito con Proyección/PSOR) y la fórmula de Black–Scholes para opciones europeas. Si QuantLib está instalado, también se muestra un precio de referencia con un motor FD.

## Cómo ejecutar (Windows, cmd)

```
cd c:\Users\tomas\Desktop\TOMI\FAMAF\to_streamlit
python -m pip install -r requirements.txt
streamlit run app.py
```

Abrí la URL local que te muestra Streamlit en el navegador.

## Notas
- QuantLib es opcional. Si no se puede instalar, el dashboard usará como referencia el promedio de PSOR.
- Numba es opcional; si está instalado acelera los solves tridiagonales/PSOR.
- Los cálculos pueden tardar para N grandes o muchas C. Ajustá N o desactivá métodos que no necesites.
