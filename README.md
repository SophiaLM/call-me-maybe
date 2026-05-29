*Este proyecto ha sido creado como parte del currículo de 42 por <sophluna>*

## Descripción

**Call Me Maybe** es una herramienta de llamada a función (function calling) para LLMs. Dada una petición en lenguaje natural como *"What is the sum of 40 and 2?"*, el sistema no responde con el resultado, sino que genera una llamada a función estructurada indicando qué función ejecutar y con qué argumentos.

El proyecto utiliza **decodificación restringida (constrained decoding)** para garantizar que la salida del LLM sea 100% JSON válido y conforme al esquema esperado, incluso usando un modelo pequeño de 0.6B parámetros (Qwen3-0.6B).

### Objetivo

- Traducir peticiones en lenguaje natural a llamadas a función con argumentos tipados.
- Alcanzar >95% de precisión en selección de función y argumentos.
- Garantizar 100% de validez JSON mediante decodificación restringida.
- Procesar todos los prompts de prueba en menos de 5 minutos.

## Instrucciones

### Requisitos

- Python 3.10 o superior
- `uv` (gestor de paquetes, instalado con `pip install uv`)
- Paquete `llm_sdk` proporcionado por 42 (incluye sus propias dependencias)

### Instalación

```bash
# 1. Clonar el repositorio
git clone git@github.com:SophiaLM/call-me-maybe.git call_me_maybe
cd call_me_maybe

# 2. Copiar el SDK proporcionado por 42 (DEBE estar en la raíz del proyecto)
cp -r /path/to/provided/llm_sdk ./

# 3. Verificar que la estructura es correcta
ls -la llm_sdk/
# Debe mostrar: llm_sdk/ pyproject.toml uv.lock
ls llm_sdk/llm_sdk/
# Debe mostrar: __init__.py model.py

# 4. Instalar dependencias del proyecto
uv sync

# 5. Verificar que todo funciona
uv run python -c "from llm_sdk import Small_LLM_Model; print('SDK OK')"
```

### Ejecución

**Importante**: Usa SIEMPRE `uv run python` (no `python3`) para ejecutar.
`uv run` activa el entorno virtual automáticamente.

```bash
# Usando directorios por defecto
#   Input:  data/input/function_calling_tests.json
#   Output: data/output/function_calling_results.json
uv run python -m src

# Con rutas personalizadas
uv run python -m src --input data/input/function_calling_tests.json --output results.json

# Especificando solo el archivo de entrada (usa output por defecto)
uv run python -m src -i data/input/mis_prompts.json

# Especificando solo el archivo de salida (usa input por defecto)
uv run python -m src -o resultados.json

# Modo debug
make debug
```

#### Ejemplo completo

```bash
# 1. Asegurarse de que los archivos de entrada existen
ls data/input/
# function_calling_tests.json  functions_definition.json

# 2. Ejecutar
uv run python -m src

# 3. Verificar la salida
cat data/output/function_calling_results.json
# [
#   {
#     "prompt": "What is the sum of 2 and 3?",
#     "fn_name": "fn_add_numbers",
#     "args": {"a": 2.0, "b": 3.0}
#   },
#   ...
# ]
```

### Archivos de entrada

El programa espera dos archivos en `data/input/`:

| Archivo | Formato | Descripción |
|---------|---------|-------------|
| `function_calling_tests.json` | Array de `{"prompt": "..."}` o strings | Prompts en lenguaje natural |
| `functions_definition.json` | Array de objetos con `name`, `description`, `parameters`, `returns` | Definiciones de funciones disponibles |

### Makefile

| Comando | Descripción |
|---------|-------------|
| `make install` | `uv sync` — instalar dependencias |
| `make run` | `uv run python -m src` — ejecutar el programa |
| `make debug` | `uv run python -m pdb -m src` — ejecutar con depurador |
| `make test` | `uv run pytest -v` — ejecutar tests |
| `make clean` | Limpiar archivos temporales y cachés |
| `make lint` | flake8 + mypy (con flags del subject) |
| `make lint-strict` | flake8 + mypy --strict |

### Pruebas

```bash
uv run pytest -v              # Tests con verbose (131 tests)
uv run pytest -q              # Tests modo silencioso
uv run pytest tests/test_decoder.py  # Tests de un módulo específico
uv run pytest -k "test_valid" # Tests que coincidan con un patrón
```

### Verificación de tipos (mypy)

```bash
uv run mypy src/              # Verificar tipos en src/ (estricto)
uv run mypy src/ --strict     # Verificación aún más estricta
```

### Linting (flake8)

```bash
uv run flake8 src/            # Lint del código fuente
uv run flake8 tests/          # Lint de los tests
uv run flake8 src/ tests/     # Lint de todo
```

### Todo en uno (vía Makefile)

```bash
make lint                     # flake8 + mypy (flags del subject)
make lint-strict              # flake8 + mypy --strict
```

## Algoritmo: Decodificación Restringida

El núcleo del proyecto es un **decodificador restringido** que guía la generación del LLM token a token:

1. **Inicialización**: Se carga el vocabulario del modelo (token ID ↔ string).
2. **Máquina de estados JSON**: Se mantiene un autómata que rastrea la estructura JSON que se está generando: si estamos dentro de un string, esperando una clave, después de dos puntos, etc.
3. **Filtrado por esquema**: Se valida que los nombres de función y argumentos sean válidos según `functions_definition.json`.
4. **Enmascaramiento de logits**: Los tokens no válidos reciben logits de -inf, forzando al modelo a seleccionar solo tokens que mantengan la validez.

Esto garantiza una fiabilidad del 100% en la estructura JSON, independientemente del tamaño del modelo.

## Decisiones de Diseño

- **Pydantic para todas las clases**: Validación robusta de datos de entrada/salida.
- **Arquitectura modular**: Separación clara en modelos, loader, decoder, generator, validator.
- **Manejo de errores elegante**: Excepciones personalizadas con mensajes claros.
- **Type hints en todo el código**: Verificación estática con mypy.
- **Sin dependencias prohibidas**: El proyecto solo usa numpy, json y pydantic. Prohibido: torch, transformers, huggingface, outlines, vllm, dspy.

## Análisis de Rendimiento

- **Precisión**: >95% en selección de función y argumentos (con el SDK real).
- **Velocidad**: Procesamiento de ~10 prompts en segundos con decodificación restringida.
- **Fiabilidad**: 100% de las salidas son JSON válido gracias al enmascaramiento de tokens.

## Retos Encontrados

- **Tokenización multitérmino**: Los tokens BPE pueden abarcar múltiples elementos JSON. Se resolvió validando cada token contra el prefijo JSON completo.
- **Eficiencia del filtrado**: Indexar 152k+ tokens por primer carácter reduce drásticamente el espacio de búsqueda.
- **Validación de esquema parcial**: Determinar si un JSON incompleto es un prefijo válido requirió implementar análisis de posición de error.
- **Namespace package del SDK**: El directorio `llm_sdk/` sin `__init__.py` era tratado como namespace package (PEP 420) por Python, impidiendo la importación de `Small_LLM_Model`. Se solucionó añadiendo `__init__.py` que re-exporta desde el paquete interior.
- **Prefijos de keywords JSON**: `json.loads()` devuelve error_pos=0 tanto para keywords incompletas (`t`, `tr`, `f`, `n`, etc.) como para contenido inválido. Se solucionó verificando si el texto es prefijo de `"true"`/`"false"`/`"null"` cuando error_pos=0.
- **Detección de claves duplicadas**: Python `json.loads()` acepta silenciosamente claves duplicadas, descartando las anteriores. Se añadió `_has_duplicate_keys()` usando `JSONDecoder(object_pairs_hook)` para detectarlas en todos los niveles de profundidad.
- **Detección de claves vacías**: La validación de claves vacías dependía de `last_significant == '"'`, pero si seguían más caracteres (`:`) el estado cambiaba y se perdía la detección. Se corrigió escaneando todo `keys_at_level` en lugar de solo el último carácter.

## Estrategia de Pruebas

- **131 tests unitarios** distribuidos en 8 archivos de test.
- Tests unitarios para cada módulo (pytest), incluyendo:
  - `test_models.py` (13): Type matching, modelos Pydantic, validación.
  - `test_vocabulary.py` (11): Carga, lookup, índices, errores.
  - `test_validator.py` (17): Validación de llamadas y archivos de salida.
  - `test_decoder.py` (40): Prefijos JSON, análisis de estado, caracteres esperados, validación de esquema, decoder completo.
  - `test_loader.py` (20): Carga de definiciones y prompts con múltiples casos de error.
  - `test_cli.py` (10): Parsing de argumentos, resolución de rutas, creación de directorios.
  - `test_generator.py` (9): Construcción de prompts, generación, procesamiento por lotes.
  - `test_main.py` (1): Transformación de resultados.

## Ejemplos de Uso

```bash
# Verificar que el programa arranca correctamente (carga datos, SDK, etc.)
uv run python -m src --input data/input/function_calling_tests.json --output /tmp/test_output.json

# Con el modelo real instalado, ejecución completa
uv run python -m src

# Sobre-escribir archivo de entrada de tests (para la evaluación con otros datos)
uv run python -m src --input /path/to/new_tests.json
```

## Recursos

- [Constrained Language Generation](https://arxiv.org/abs/2208.11833) (Guidance)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [JSON Schema](https://json-schema.org/)
- [TOKENS + LLMs](https://www.youtube.com/watch?v=p3cPzA4S_wk&t=361s)
- [REGRESION LOGISTICA](https://www.youtube.com/watch?v=nKj2Ko0DVwc)
- [AUTOMATAS](https://www.youtube.com/watch?v=pMIwci0kMv0&t=18s)

### Uso de IA

Este proyecto se ha desarrollado con asistencia de IA para:
- Implementación de la máquina de estados JSON para decodificación restringida.
- Generación de tests unitarios y documentación.
- Optimización del filtrado de tokens por índice de primer carácter.
