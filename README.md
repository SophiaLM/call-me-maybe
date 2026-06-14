*Este proyecto ha sido creado como parte del currículo de 42 por sophluna*

## Descripción

**call me maybe** es una herramienta de *function calling* que traduce peticiones en lenguaje natural en llamadas a funciones estructuradas con argumentos tipados. Utiliza *constrained decoding* para garantizar que la salida sea un JSON 100% válido y conforme al esquema definido.

Dada una pregunta como *"What is the sum of 40 and 2?"*, el sistema no responde *"42"* sino que produce:

```json
{
  "fn_name": "fn_add_numbers",
  "args": {"a": 40, "b": 2}
}
```

La implementación se apoya en un modelo de lenguaje pequeño (Qwen3-0.6B) y guía la generación token a token mediante una máquina de estados que solo permite tokens que mantengan la validez sintáctica y semántica del JSON de salida.

### Requisitos

- Python 3.10+
- uv (gestor de paquetes)

### Instalación

```bash
uv sync
```

### Ejecución

```bash
make run
```

O con rutas personalizadas:

```bash
uv run python -m src --input data/input/ --output data/output/
```

### Makefile

| Comando | Descripción |
|---------|-------------|
| `make install` | Instalar dependencias |
| `make run` | Ejecutar el programa |
| `make debug` | Ejecutar con pdb |
| `make clean` | Limpiar archivos temporales |
| `make lint` | flake8 + mypy |

## Explicación del algoritmo de constrained decoding

El *constrained decoding* (decodificación restringida) es una técnica que modifica la generación token a token de un LLM para garantizar que la salida siga una estructura predefinida.

### Funcionamiento paso a paso

1. **El modelo produce logits**: En cada paso de generación, el LLM asigna una puntuación (logit) a cada token de su vocabulario.

2. **Identificar tokens válidos**: Según la posición actual en la estructura JSON (¿esperamos una clave? ¿un valor numérico? ¿el cierre del objeto?), se determina qué tokens mantendrían la validez.

3. **Enmascarar logits inválidos**: Todos los tokens que romperían la estructura tienen sus logits fijados a -inf, haciéndolos imposibles de seleccionar.

4. **Seleccionar el mejor token**: Se elige el token con el logit más alto entre los válidos (argmax).

5. **Avanzar la máquina de estados**: El token elegido actualiza el estado del parser, determinando los tokens válidos para el siguiente paso.

### Dos fases de generación

- **Selección de función**: El LLM elige qué función llamar. El constraint solo permite tokens que formen parte de los nombres de función disponibles.

- **Generación de argumentos**: El LLM genera un JSON con los argumentos. El constraint verifica que las claves sean nombres de parámetros válidos y que los valores tengan el tipo correcto (number, string o boolean).

### Máquina de estados JSON

```
START → KEY_OR_CLOSE → IN_KEY → AFTER_KEY → BEFORE_VALUE
         ↓                             ↓
        DONE                    IN_STRING / IN_NUMBER / IN_TRUE / IN_FALSE
                                     ↓               ↓
                                AFTER_VALUE ←────────┘
                                     ↓
                              KEY_OR_CLOSE / DONE
```

Cada estado define qué tokens son válidos. Por ejemplo, en estado `IN_NUMBER` solo se permiten dígitos, punto decimal, coma (siguiente parámetro) o llave de cierre.

## Decisiones de diseño

| Decisión | Alternativa | Motivo |
|----------|-------------|--------|
| Máquina de estados explícita vs. regex | La máquina de estados permite rastrear posición exacta y tipos de parámetros | Necesaria para validar tipos por parámetro |
| Dos fases separadas (selector + args) | Generar todo el JSON en un solo pase | La selección de función necesita un constraint distinto (nombres válidos) |
| argmax (greedy) vs. sampling | Muestreo podría dar más variedad | argmax es determinista y más fiable para estructura |
| Pydantic para todos los modelos | Validación manual con dicts | Pydantic garantiza validación automática y errores claros |

## Análisis de rendimiento

### Precisión

Con el modelo de prueba (mock), la selección de función queda determinada por la frecuencia de letras del vocabulario. Con un modelo real (Qwen3-0.6B), la precisión semántica dependería de la capacidad del modelo para entender el prompt.

La precisión estructural es del **100%**: todo el JSON generado es sintácticamente válido y cumple el esquema.

### Velocidad

Cada prompt requiere dos pases de generación (selector + args). Con el modelo mock, cada generación completa ~10-50 tokens en <1 segundo. El límite de 5 minutos para todos los prompts se cumple ampliamente.

### Fiabilidad

El constrained decoding garantiza que:
- `fn_name` siempre es un nombre de función válido
- `args` siempre incluye todos los parámetros requeridos
- Los tipos de los argumentos coinciden con la definición

## Retos encontrados

1. **Nombres de archivo**: El plan de acción referencia `function_definitions.json` pero el archivo real se llama `functions_definition.json`. Se implementó detección de ambos nombres.

2. **Formato de tests**: Los prompts de prueba pueden ser strings planas u objetos con clave `"prompt"`. El loader soporta ambos formatos.

3. **Tokens multi-caracter**: El modelo mock usa un vocabulario de un carácter por token, simplificando el mapeo. El diseño del constraint es compatible con tokens multi-caracter si se usa un modelo real.

4. **Espacios en blanco**: El modelo mock asigna alta puntuación al espacio, lo que podía generar espacios en posiciones inesperadas. El constraint ignora espacios (no cambian el estado).

## Estrategia de pruebas

Las pruebas se ejecutan directamente con los archivos en `data/input/`:

```bash
uv run python -m src
```

Casos de prueba incluidos:

- **Operaciones básicas**: suma, saludo, reversión de string, raíz cuadrada, sustitución regex
- **Casos límite**: prompt vacío, solo espacios, números negativos, cero, valores grandes
- **Caracteres especiales**: strings con puntuación y números
- **Nombres largos**: parámetros de string con longitud extrema

Validación post-ejecución:

```bash
python -c "import json; json.load(open('data/output/function_calling_results.json'))"
```

## Recursos

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [uv Package Manager](https://docs.astral.sh/uv/)
- [Constrained Decoding (Lily Weng)](https://lilyweng.notion.site/Constrained-Decoding-for-LLMs-3561e139d9e14cb78e0b7e96d1d2921f)

### Uso de IA

Este proyecto ha sido desarrollado con asistencia de IA (opencode) para:
- Generación del esqueleto del código siguiendo el plan de acción
- Debugging de tipos y errores de linting
- Documentación y este README
