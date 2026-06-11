# Guía del Proyecto: call_me_maybe

## Lógica Seguida para Crear el Proyecto

### 1. Análisis de la Documentación (subject.pdf)

Se extrajeron todos los requisitos del PDF:
- **Lenguaje**: Python 3.10+ con type hints.
- **Estilo**: flake8 + mypy con flags específicos.
- **Clases**: Todas con Pydantic para validación.
- **Dependencias**: numpy, json, pydantic. Prohibido: torch, transformers, huggingface, outlines, vllm, dspy.
- **Paquetes**: uv para gestión de dependencias. `uv sync` para instalar.
- **Modelo**: Qwen/Qwen3-0.6B (por defecto).
- **Makefile**: install, run, debug, clean, lint, lint-strict.
- **Selección de función**: Debe hacerla el LLM, no heurísticas.
- **Decodificación restringida**: No depender del prompt para obtener JSON válido.
- **llm_sdk**: Paquete proporcionado, copiado al directorio del proyecto.

### 2. Arquitectura del Proyecto

```
call_me_maybe/
├── src/
│   ├── __init__.py        # Package marker
│   ├── __main__.py        # Entry point (orquestación)
│   ├── cli.py             # Parsing de argumentos CLI
│   ├── models.py          # Modelos Pydantic
│   ├── errors.py          # Excepciones personalizadas
│   ├── loader.py          # Carga de archivos de entrada
│   ├── vocabulary.py      # Mapeo token ID ↔ string
│   ├── decoder.py         # Decodificación restringida (core)
│   ├── generator.py       # Interacción con el LLM
│   └── validator.py       # Validación de salida
├── llm_sdk/               # SDK del modelo (proporcionado)
│   ├── __init__.py
│   └── model.py           # Small_LLM_Model (mock + interfaz)
├── data/
│   ├── input/
│   │   ├── function_calling_tests.json
│   │   └── functions_definition.json
│   └── output/            # Generado en ejecución
├── tests/
│   ├── __init__.py
│   ├── test_models.py     # Tests de modelos Pydantic
│   ├── test_decoder.py    # Tests del core de decodificación
│   ├── test_loader.py     # Tests de carga de archivos
│   ├── test_validator.py  # Tests de validación
│   ├── test_vocabulary.py # Tests de vocabulario
│   ├── test_cli.py        # Tests de CLI
│   ├── test_generator.py  # Tests de generación
│   ├── test_main.py       # Tests del entry point
├── .flake8                # Config de flake8
├── pyproject.toml
├── Makefile
├── .gitignore
├── README.md
└── guia.md
```

### 3. Flujo de Ejecución

```
CLI args → Loader (carga JSON) → Vocabulary (carga token→string)
    ↓
Generator (construye prompt con definiciones de funciones)
    ↓
ConstrainedDecoder (bucle token a token):
    ├── Codificar texto actual → obtener logits del LLM
    ├── Determinar estado JSON actual
    ├── Filtrar tokens válidos (JSON + esquema)
    ├── Enmascarar tokens inválidos (logits = -inf)
    ├── Seleccionar token (argmax)
    └── Repetir hasta completar JSON
    ↓
Validator (valida JSON de salida)
    ↓
Build results → Escribir output JSON
```

### 4. Decodificación Restringida (Algoritmo Core)

El algoritmo de decodificación restringida funciona en 4 pasos por cada token generado:

1. **Análisis de estado**: Dado el texto parcial generado, se analiza:
   - ¿Estamos dentro de un string?
   - ¿Cuál es la profundidad de llaves `{}`?
   - ¿Cuál fue el último carácter significativo?
   - ¿Qué claves se han visto hasta ahora?
   - ¿Qué valores se han completado?

2. **Determinación de caracteres válidos**: Basado en el estado:
   - `{` → esperamos `"`
   - `"` (cerrando key) → esperamos `:`
   - `:` → esperamos valor (`"`, `{`, `[`, dígito)
   - `,` → esperamos `"` (nueva clave)
   - `}` → esperamos `,` o `}`
   - Dentro de string de función: solo caracteres que sean prefijo de un nombre de función válido
   - Dentro de string de argumento: solo caracteres que sean prefijo de un parámetro existente
   - Después de `:` para parámetro numérico: excluye `"`, `{`, `[`

3. **Filtrado de tokens**: Se buscan en el vocabulario los tokens cuyo primer carácter coincida con los esperados, pre-indexados para eficiencia O(1).

4. **Validación de esquema**: Cada token candidato se evalúa simulando su adición:
   - ¿Sigue siendo el resultado un prefijo JSON válido? (`_is_valid_json_prefix`)
   - ¿Es compatible con el esquema de la función? (`_matches_function_schema`)

### 5. Tipos de Validación

El sistema tiene dos capas de validación:

**Capa 1 — `_get_expected_first_chars`:**
- Determina los primeros caracteres estructuralmente válidos según el estado JSON actual.
- Para claves en depth 1: fuerza el orden "function" antes de "arguments".
- Para nombres de función: solo caracteres que sean prefijo de una función registrada.
- Para parámetros: solo caracteres que sean prefijo del siguiente parámetro disponible.
- Para valores numéricos: excluye `"`, `{`, `[` cuando el parámetro es "number"/"integer".

**Capa 2 — `_validate_incomplete_schema`:**
- Defense-in-depth que valida el texto parcial completo.
- Verifica claves vacías, claves duplicadas, orden de claves, nombres de función válidos, nombres de parámetros válidos, tipos de valores.
- Corre después de `_is_valid_json_prefix` para atrapar cualquier caso que `_get_expected_first_chars` no cubra.

### 6. Módulo de Vocabulary

El archivo `vocabulary.py` carga el JSON de vocabulario del modelo (token_id → string) y construye:
- `id_to_token`: Dict[int, str] - lookup directo.
- `token_to_ids`: Dict[str, List[int]] - búsqueda inversa.
- `by_first_char`: Dict[str, List[int]] - índice por primer carácter para filtrado rápido.

## Explicación Detallada de Cada Archivo

### `src/__init__.py`
Archivo vacío que marca `src/` como paquete Python.

### `src/__main__.py`
Punto de entrada `python -m src`.
- `main()`: Orquesta todo el flujo:
  1. Parsea argumentos CLI con `parse_args()`.
  2. Resuelve rutas con `resolve_paths()`.
  3. Carga definiciones de funciones y prompts con `load_function_definitions()` y `load_prompts()`.
  4. Inicializa el modelo LLM (`Small_LLM_Model`) y vocabulario (`Vocabulary`).
  5. Genera llamadas a función con `process_prompts()`.
  6. Convierte resultados al formato de salida con `build_results()`.
  7. Escribe el archivo JSON de salida.
- `build_results()`: Transforma `FunctionCallOutput` (function, arguments) al formato final (prompt, fn_name, args). Maneja type casting para booleanos.
- Manejo de errores con try-except: `LoaderError`, `VocabularyError`, `GeneratorError`, `CallMeMaybeError`, `NotImplementedError`.

### `src/cli.py`
- `parse_args(argv=None)`: Define argumentos `--input`/`-i` y `--output`/`-o` usando argparse.
- `resolve_paths(args)`: Convierte argumentos en rutas:
  - Si `--input` se proporciona, deriva el directorio de entrada del path.
  - Si no, usa `data/input/` por defecto.
  - El directorio de salida se crea automáticamente si no existe.

### `src/models.py`
Funciones y modelos Pydantic:
- `type_matches(value, expected_type)`: Verifica tipos Python contra tipos JSON (string, number, boolean, integer, object, array, null). Función pura sin dependencias externas.
- `ParameterDefinition`: type del parámetro (ej: "string", "number").
- `ReturnDefinition`: type del retorno.
- `FunctionDefinition`: name, description, parameters (Dict[str, ParameterDefinition]), returns (Optional[ReturnDefinition]).
- `PromptEntry`: prompt individual (wrapper de string).
- `FunctionCallOutput`: salida del LLM (function, arguments). Incluye `model_validator` que rechaza nombre de función vacío.

### `src/errors.py`
Jerarquía de excepciones:
- `CallMeMaybeError`: base de todas las excepciones del proyecto.
- `VocabularyError`: errores de carga/lookup de vocabulario.
- `DecoderError`: errores de decodificación restringida.
- `LoaderError`: errores de carga/parseo de archivos.
- `GeneratorError`: errores durante la generación del LLM.
- `ValidationError`: errores de validación de salida.

### `src/loader.py`
- `load_function_definitions(input_dir)`: Busca el archivo de definiciones probando múltiples nombres (`functions_definition.json`, `function_definitions.json`, `functions.json`). Valida estructura (lista de objetos), nombres únicos, y parsea con Pydantic (`FunctionDefinition`).
- `load_prompts(input_dir)`: Carga el archivo `function_calling_tests.json`. Soporta tanto arrays de strings como de objetos `{"prompt": "..."}`. Valida contra `PromptEntry` de Pydantic.

### `src/vocabulary.py`
- `Vocabulary.__init__(path)`: Carga el JSON de vocabulario. Construye índices: `id_to_token`, `token_to_ids`, `by_first_char`.
- `_load(path)`: Lee y parsea el archivo JSON. Valida que las claves sean enteros válidos.
- `token_to_string(token_id)`: Obtener string de un token ID (O(1)).
- `string_to_token_ids(text)`: Obtener todos los IDs para un string exacto.
- `get_tokens_by_first_char(char)`: Obtener tokens que empiezan con un carácter (O(1) lookup en índice).
- `get_tokens_by_prefix(prefix)`: Obtener tokens que empiezan con un prefijo (filtra por primer carácter + verificación completa).
- `all_token_ids()`: Lista de todos los token IDs.
- `size()`: Número de tokens en el vocabulario.

### `src/decoder.py` (Core del Proyecto)

**Funciones auxiliares**:
- `_is_valid_json_prefix(text)`: Determina si un string es prefijo JSON válido. Estrategia:
  1. Verifica que el primer carácter sea válido (`{`, `"`, `[`, dígito, `t`, `f`, `n`, `-`).
  2. Intenta `json.loads()` — si funciona, es JSON completo válido.
  3. Si falla, analiza `JSONDecodeError.pos`:
     - Error al final → incompleto → prefijo válido.
     - Error antes del final → error de sintaxis → prefijo inválido.
     - `error_pos == 0` → verifica si es prefijo de keyword JSON (`true`/`false`/`null`) o número negativo (`-`).
  4. Casos especiales: strings abiertos, escapes, caracteres estructurales.
- `_get_partial_prefix_state(text)`: Analiza el texto parcial carácter por carácter. Retorna:
  - `in_string`: ¿Estamos dentro de un string?
  - `escaped`: ¿Estamos en modo escape?
  - `brace_depth`: profundidad actual de `{}`.
  - `last_significant`: último carácter no-whitespace.
  - `current_key`: clave actual en construcción (si in_string y key).
  - `keys_at_level`: mapeo depth → lista de claves completadas.
  - `after_colon`: ¿Acabamos de ver `:`?
  - `reading_value`: ¿Estamos leyendo un valor (después de `:` y `"`)?
  - `last_key_at_depth`: mapeo depth → última clave leída.
  - `completed_values`: mapeo depth → dict de clave→valor completados.
- `_get_expected_first_chars(state, fn_map)`: Dado el estado, devuelve los caracteres válidos para el siguiente token. Lógica principal:
  - En strings: restringe primeros caracteres según contexto (clave vs valor, depth, función vs argumento).
  - Fuera de strings: determina caracteres según `last_significant`.
  - Para funciones y parámetros: solo permite caracteres que sean prefijo de nombres válidos.
  - Para parámetros numéricos: excluye `"` (no se permiten strings).
  - Cuando no quedan parámetros: elimina `,` para forzar cierre.
- `_has_duplicate_keys(text)`: Detecta claves duplicadas en JSON usando `json.JSONDecoder(object_pairs_hook=hook)`. El hook se ejecuta en cada nivel del JSON, detectando duplicados a cualquier profundidad.
- `_matches_function_schema(text, fn_map)`: Verifica que el texto generado sea compatible con el esquema de funciones:
  1. Primero verifica duplicados con `_has_duplicate_keys`.
  2. Si el JSON es completo, usa `_validate_complete_schema`.
  3. Si es incompleto, usa `_validate_incomplete_schema`.
- `_validate_complete_schema(obj, fn_map)`: Valida un objeto JSON completo:
  - Verifica que las claves sean solo "function" y "arguments".
  - Valida que el nombre de función exista en fn_map.
  - Valida que los argumentos sean un objeto con claves/tipos correctos.
- `_validate_incomplete_schema(text, fn_map)`: Valida un string JSON incompleto. Estrategia:
  1. Parsea objetos JSON completos del texto con `raw_decode`.
  2. Analiza el estado con `_get_partial_prefix_state`.
  3. Verifica (en orden):
     - (1) No claves vacías en ningún nivel.
     - (2) Claves en depth 1 deben ser prefijo de "function"/"arguments".
     - (2b) "function" debe preceder a "arguments".
     - (3) Claves en depth 2 deben ser prefijo de parámetros válidos.
     - (3.5) Valor de "function" debe ser prefijo de nombre de función válido.
     - (4) Valor de "arguments" no puede ser string.
     - (5) Claves completadas en depth 1 deben ser válidas y "function" antes de "arguments".
     - (6) Claves completadas en depth 2 deben ser parámetros válidos, sin duplicados.
     - (6.5) No claves duplicadas en ningún nivel.
     - (7) Nombre de función completado debe existir en fn_map.
     - (8) Argumentos completados deben ser parámetros válidos de la función.

**Clase `ConstrainedDecoder`**:
- `__init__(function_defs, eos_token_id)`: Almacena definiciones de funciones (Dict[name → FunctionDefinition]) y el ID del token EOS.
- `set_vocabulary(vocab)`: Configura el vocabulario (actualmente no hace nada extra).
- `reset()`: Reinicia `self.partial` a string vacío.
- `get_valid_token_ids(logits, vocab)`: Enmascara logits para solo permitir tokens válidos:
  1. Obtiene estado actual del texto parcial con `_get_partial_prefix_state`.
  2. Determina primeros caracteres esperados con `_get_expected_first_chars`.
  3. Si no hay caracteres esperados: permite solo EOS.
  4. Obtiene tokens candidatos por primer carácter desde el vocabulario.
  5. Valida cada candidato contra `_is_valid_json_prefix` + `_matches_function_schema`.
  6. Si ningún candidato es válido: permite solo EOS.
  7. Retorna logits enmascarados (solo tokens válidos mantienen su logit original).
- `_is_complete_call(text)`: Verifica si el texto es una llamada a función JSON completa: parseable con `json.loads`, tiene "function" y "arguments", y brace_count == 0.
- `step(token_id, token_str)`: Avanza el decoder con un token generado:
  - Si el texto ya es una llamada completa → return False.
  - Si el token es EOS → return False.
  - Si no: concatena el token, verifica si ahora está completa, verifica longitud máxima.

### `src/generator.py`
Interacción con el LLM y generación de llamadas a función.
- `build_prompt(prompt_entry, fn_map)`: Construye el prompt del sistema con las definiciones de funciones en JSON. Template con `{FUNCTIONS_JSON}` y `{USER_PROMPT}`.
- `generate_function_call(model, prompt, decoder, vocab, max_new_tokens)`: Bucle de generación token a token:
  1. Codifica el prompt completo con `model.encode()`.
  2. Reinicia el decoder con `decoder.reset()`.
  3. Por cada paso: obtiene logits del modelo, enmascara con decoder, selecciona argmax.
  4. Si el token es EOS o decoder dice "completo": termina.
  5. Retorna `decoder.partial` (el JSON generado).
  6. Errores: `GeneratorError` si falla encode, logits, o generación vacía.
- `process_prompts(model, prompts, fn_defs, vocab, decoder)`: Procesa todos los prompts:
  - Por cada prompt: construye prompt completo, genera, valida.
  - Retorna lista de (PromptEntry, FunctionCallOutput).

### `src/validator.py`
- `validate_function_call_json(json_str, fn_map)`: Parsea y valida el JSON generado por el LLM:
  - Verifica que no esté vacío.
  - Parsea con `json.loads()`.
  - Verifica que sea un objeto.
  - Verifica que tenga "function" y "arguments" (y solo esas claves).
  - Valida que "function" sea string y nombre conocido.
  - Valida que "arguments" sea objeto.
  - Valida cada argumento: clave conocida, tipo correcto.
  - Retorna `FunctionCallOutput`.
  - Errores: `ValidationError` con mensajes descriptivos.
- `validate_output_file(path)`: Valida el archivo JSON de salida completo:
  - Verifica que el archivo exista y sea JSON válido.
  - Verifica que sea un array no vacío.
  - Valida cada entrada: prompt (string), fn_name (string), args (object).

### `llm_sdk/` (Mock del SDK)
- `Small_LLM_Model.__init__(model_dir, device)`: Toma un directorio de modelo (o usa "mock" por defecto). Prepara caché de vocabulario.
- `get_path_to_vocabulary_json()`: Busca `vocab.json` o `vocabulary.json` en el directorio del modelo. Si no encuentra, genera un vocabulario por defecto (token 0=`<unk>`, 1=`<s>`, 2=`</s>`, 3-97 = ASCII imprimible) y lo guarda en un archivo temporal.
- `get_logits_from_input_ids(input_ids)`: Genera logits simulados. Cada token recibe una puntuación basada en la frecuencia de su primer carácter (letras comunes = alta, símbolos JSON = alta, caracteres raros = baja). Independiente del input (mock).
- `encode(text)`: Codifica texto carácter por carácter. Cada carácter se mapea a su token ID según el vocabulario (o a `<unk>`=0 si no existe).
- `decode(token_ids)`: Decodifica IDs a string concatenando las representaciones de cada token.
- `get_vocab_dict()`: Retorna el vocabulario como Dict[str, str].
- Propiedades: `eos_token_id` (siempre 2), `vocab_size`.

### `tests/` (131 tests en 8 archivos)
- `test_models.py` (13 tests): `TestTypeMatches` (8 tipos), `TestParameterDefinition`, `TestReturnDefinition`, `TestFunctionDefinition` (3 casos: mínimal, con params, con returns), `TestPromptEntry`, `TestFunctionCallOutput` (2 casos: válido, nombre vacío).
- `test_decoder.py` (40 tests): `TestIsValidJsonPrefix` (9 casos: vacío, whitespace, JSON completo, incompleto, inválido, string abierto, keyword prefix, numérico), `TestGetPartialPrefixState` (8 estados), `TestGetExpectedFirstChars` (10 posiciones), `TestValidateCompleteSchema` (6 casos), `TestValidateIncompleteSchema` (9 casos), `TestMatchesFunctionSchema` (3 casos), `TestConstrainedDecoder` (7 casos: init, reset, valid tokens, step, complete call, EOS, límite de longitud).
- `test_loader.py` (20 tests): `TestLoadFunctionDefinitions` (10 casos: 3 nombres de archivo, archivo faltante, JSON inválido, no-lista, entrada no-dict, sin nombre, duplicado, vacío, directorio faltante), `TestLoadPrompts` (10 casos: carga normal, archivo faltante, JSON inválido, no-lista, strings, dict sin prompt, tipo inesperado, vacío).
- `test_validator.py` (17 tests): `TestValidateFunctionCallJson` (11 casos: válido, vacío, sintaxis inválida, no-objeto, claves faltantes, claves extra, function no-string, función desconocida, arguments no-objeto, argumentos extra, tipo incorrecto), `TestValidateOutputFile` (6 casos: archivo faltante, JSON inválido, válido, no-lista, array vacío, entrada con claves faltantes).
- `test_vocabulary.py` (11 tests): `TestVocabulary` (11 casos: carga, token→string, string→tokens, primer carácter, prefijo, prefijo vacío, todos los IDs, repr, archivo inexistente, archivo vacío, JSON inválido).
- `test_cli.py` (10 tests): `TestParseArgs` (6 casos: defaults, input largo, input corto, output largo, output corto, ambos), `TestResolvePaths` (4 casos: defaults, input custom deriva dir, output custom, creación de directorio de salida).
- `test_generator.py` (9 tests): `TestBuildPrompt` (2 casos: contenido del prompt), `TestGenerateFunctionCall` (2 casos: generación válida, error en encode), `TestProcessPrompts` (1 caso: procesamiento completo).
- `test_main.py` (1 test): `TestBuildResults` (transformación de resultados con type casting).

## Los 6 Errores Más Complejos y su Solución

### Error 1: JSON válido pero incompleto como prefijo

**Problema**: `json.loads()` lanza `JSONDecodeError` para todo JSON incompleto, pero no todos los JSON incompletos son prefijos válidos. Había que distinguir entre "incompleto pero válido" (ej: `{"function"`) y "sintácticamente inválido" (ej: `{"function" 1}`).

**Solución**: Se implementó `_is_valid_json_prefix()` que:
1. Intenta `json.loads()` → si funciona, es JSON completo válido.
2. Si falla, analiza `JSONDecodeError.pos`:
   - Si el error está al final del string → es incompleto → prefijo válido.
   - Si está antes del final → error de sintaxis → prefijo inválido.
3. Casos especiales: strings abiertos, después de caracteres estructurales (`{`, `:`, `,`), etc.

### Error 2: Tokens multi-carácter que cruzan límites JSON

**Problema**: Los tokenizadores pueden tener tokens que combinan múltiples elementos JSON. Por ejemplo, un solo token podría ser `"function":` (incluyendo key, colon, y espacio). El filtrado por "primer carácter esperado" sería insuficiente si el primer carácter es válido pero el resto del token rompe la estructura.

**Solución**: Validación completa para cada token candidato:
1. Se obtienen todos los tokens cuyo primer carácter coincide con los esperados.
2. Para cada uno, se simula la concatenación: `partial + token_str`.
3. Se valida el resultado completo contra `_is_valid_json_prefix()` y `_matches_function_schema()`.
4. Solo los tokens que pasan ambas validaciones se mantienen.

### Error 3: Formato de salida LLM vs archivo final

**Problema**: Inicialmente se confundió el formato de salida del LLM con el formato del archivo de salida. El LLM debe generar `{"function": "fn_name", "arguments": {...}}` y luego el programa lo convierte a `{"prompt": "...", "fn_name": "...", "args": {...}}`.

**Solución**: Separación clara de responsabilidades:
- `generator.py` genera `FunctionCallOutput` (con `function` y `arguments` del LLM).
- `__main__.py` transforma al formato final con `build_results()` (con `prompt`, `fn_name`, `args`).
- `validator.py` tiene dos funciones: `validate_function_call_json()` para la salida del LLM y `validate_output_file()` para el archivo final.

### Error 4: Prefijos de keywords JSON (t, tr, f, n, etc.)

**Problema**: `json.loads("t")` devuelve `JSONDecodeError` con `pos=0` porque la keyword "true" está incompleta. Lo mismo para `"tr"`, `"tru"`, `"f"`, `"fa"`, `"n"`, `"nu"`, `"nul"`. El código original trataba `error_pos == 0` como error de sintaxis, pero en realidad es un prefijo válido de keyword JSON.

**Solución**: En `_is_valid_json_prefix()`, cuando `error_pos == 0`, se verifica si el texto es prefijo de alguna keyword JSON (`"true"`, `"false"`, `"null"`) o si es un número negativo incompleto (`"-"`). Si es así, se retorna True.

```python
if error_pos == 0:
    if any(kw.startswith(stripped) for kw in ["true", "false", "null"]):
        return True
    if stripped == "-":
        return True
```

### Error 5: Claves duplicadas en JSON

**Problema**: `json.loads()` en Python acepta silenciosamente claves duplicadas en JSON, conservando solo el último valor. Esto significa que `{"a": 1, "a": 2}` se parsea como `{"a": 2}` sin error. El decoder podía generar JSON con claves duplicadas sin ser detectado.

**Solución**: Dos capas de defensa:
1. `_has_duplicate_keys()`: Usa `json.JSONDecoder(object_pairs_hook=hook)` que llama al hook con los pares (key, value) de cada nivel del JSON. Si en algún nivel hay claves duplicadas, se detecta inmediatamente.
2. En `_validate_incomplete_schema()`: Se añadió el check `(6.5)` que verifica duplicados en `keys_at_level` para cualquier profundidad.

### Error 6: Claves vacías no detectadas después de `:`

**Problema**: La validación de claves vacías (check 1) originalmente solo se activaba cuando `last_significant == '"'`, que es el momento inmediatamente después de cerrar una clave. Pero si seguían más caracteres (como `:`), `last_significant` cambiaba y la clave vacía pasaba desapercibida.

**Solución**: Se cambió la detección de claves vacías para escanear TODOS los valores de `keys_at_level` en lugar de depender de `last_significant`:

```python
for keys in state.get("keys_at_level", {}).values():
    if any(k == "" for k in keys):
        return False
```

## Información Adicional

### Formato del Archivo de Vocabulario

El archivo de vocabulario es un JSON donde las claves son IDs de token (strings) y los valores son las representaciones textuales:

```json
{
  "0": "<unk>",
  "1": "<s>",
  "2": "</s>",
  "100": "hello",
  "101": " world",
  "892": "What",
  "4771": "sum"
}
```

### Vocabulario por defecto (mock)

Cuando no hay un archivo de vocabulario real, `Small_LLM_Model._build_default_vocab()` genera uno con:
- Token 0: `<unk>` (desconocido)
- Token 1: `<s>` (start)
- Token 2: `</s>` (end)
- Tokens 3-97: caracteres ASCII imprimibles (códigos 32-126), un carácter por token.

Esto significa que en modo mock, cada carácter es un token individual (tokenización carácter a carácter). Esto simplifica el constrained decoding porque no hay tokens multi-carácter que puedan cruzar límites JSON.

### Sistema de Puntuación Mock

El mock `Small_LLM_Model` asigna logits basados en la frecuencia del primer carácter del token:
- Caracteres JSON estructurales (`"`, `{`, `}`, `:`, `,`): 3.0-3.5
- Letras comunes (e, t, a, o, i, n, s): 1.9-2.8
- Letras medias (h, r, d, l, c, u, m): 1.2-1.9
- Letras raras (w, f, g, y, p, b, v, k, j, x, q, z): 0.1-1.1
- Dígitos: 0.2-2.0
- Caracteres especiales: -5.0 (por defecto)
- Token vacío: -10.0

Esto asegura que el decoder pueda generar JSON válido incluso con un modelo que no atiende al prompt.

### .flake8 vs pyproject.toml

Aunque el proyecto usa `pyproject.toml` para la mayoría de la configuración, flake8 no siempre lee correctamente la sección `[tool.flake8]` (varía según la versión). Por eso se incluye un archivo `.flake8` separado con la misma configuración:
```
[flake8]
max_line_length = 88
extend-ignore = E501, W503
exclude = .venv, venv, __pycache__, .mypy_cache, .pytest_cache
```

### Comandos de verificación

```bash
# Ejecutar el proyecto
uv run python -m src

# Tests
uv run pytest -v                    # 131 tests
uv run pytest -q                    # Modo silencioso

# Type checking
uv run mypy src/                    # Solo src/ (estricto)
uv run mypy src/ tests/             # Incluyendo tests

# Linting
uv run flake8 src/ tests/           # Lint completo

# Lint + mypy via Makefile
make lint
make lint-strict
```

### Manejo de Errores

El proyecto maneja elegantemente:
- Archivos JSON mal formados (JSONDecodeError → LoaderError/ValidationError).
- Archivos no encontrados (con mensajes de error claros y rutas mostradas).
- Valores de argumentos con tipos incorrectos (type_matches → ValidationError).
- Nombres de función inválidos (ValidationError con lista de funciones válidas).
- Tokens desconocidos en el vocabulario (return empty string).
- Directorios de entrada/salida inexistentes (creación automática o error descriptivo).
- Fallo en codificación/generación del LLM (GeneratorError con causa original).
- Claves duplicadas en JSON (rechazadas por _has_duplicate_keys).
- Claves vacías en JSON (rechazadas por check 6.5).

### Seguridad

- No se exponen API keys ni secretos.
- No se evalúa código generado (no `exec`/`eval`).
- Las rutas de archivos se validan antes de usar.
- Las importaciones están auditadas (no torch/transformers/etc.).
- La generación de JSON es 100% controlada por el decoder (no hay riesgo de inyección).
