# RunPod: перенести next-moe з Colab

Файли готові, але ця cloud environment не має доступу до вашого RunPod account.
Pod створюється користувачем у UI; GPU billing не запускається нашими скриптами.

## 1. Зберегти й перенести Colab

Після завершення/Stop training cell виконайте в Colab:

```python
import urllib.request
exec(urllib.request.urlopen(
    'https://raw.githubusercontent.com/FactorialD/next-moe/work/scripts/runpod_transfer.py'
).read().decode('utf-8'))
bundle = make_runpod_bundle(trainer, include_optimizer=False)
from google.colab import files
files.download(str(bundle))
```

Default ZIP≈0.8GiB, створюється локально /content, не дублюється наDrive:
weights.pt + data-replay/ + phase.json, доступні hashes/baseline/metrics/pip-freeze.
Передається поточний trainer, не довільно старий run. Tokenizer/holdouts незмінні.
ZIP має transfer-manifest.json із SHA кожного файлу. Це hashes цілісності,
не цифровий підпис; використовуйте власний archive.

Для exact migration поставте include_optimizer=True: додається full last.pt≈3.83GiB,
ZIP загалом≈4.6GiB, викликається trainer.save(). Потрібне місце наColab localdisk.
З компактним ZIP Adam/RNG reset і warm ramp; exact переносить masters/Adam/RNG.
Не завершувати Colab runtime, доки ZIP не завантажився накомп'ютер; уже збережений
Drive checkpoint залишити резервною копією. Один і той самий ZIP filename не
перезаписується автоматично: задайте інший destination або видаліть старий ZIP.
Не упаковуємо всі попередні run, CUDA env чи весь Drive.

## 2. Створити GPU Pod

- Pods → Deploy Pod, **1 GPU**. Перший короткий тест: RTX2000Ada16GB за ціною,
  показаною вам ($0.24/hr на вашому screenshot). Поточну ціну перевіряти у UI.
  Це кандидат на економічність, speed заздалегідь невідомий. Наступний кандидатL40S.
- Шаблон **RunPod PyTorch**, PyTorch>=2.6 із CUDA та Jupyter. Для Ada CUDA12.4+
  є практичною основою; не встановлювати torch CPU wheel поверхCUDAtemplate.
- Container disk20GB; постійний volume20GB, mount /workspace. Якщо доступний
  Network Volume у потрібному region/GPU, його можна повторно приєднувати після
  завершенняPod. Звичайний volume disk зберігається при Stop того самогоPod,
  але перед Terminate потрібна зовнішня резервна копія. ДоступністьGPU залежить
  від region та storage selection. Перевірити актуальні RunPod storage terms.
- Підсумкова GPU price та storage fees оплачуються окремо. Починайте з одногоPod.
  Spot дешевший, але може бути перерваний; перший setup простіший наOn-Demand.
- Deploy → Connect → Jupyter. Використовувати авторизований Connect link,
  не відкривати Jupyter без token/password. Додаткові public ports не потрібні.

Довідка сервісу: https://docs.runpod.io/ . У цьому сеансі Pod не створювався,
фактичні mount/price/CUDA/Jupyter навашомуPod потребують перевірки після запуску.

## 3. Terminal у Jupyter: код і dependencies

```bash
cd /workspace
git clone --branch work https://github.com/FactorialD/next-moe.git
bash /workspace/next-moe/scripts/setup_runpod.sh
```

Якщо checkout уже є, не клонувати поверх: використовуйте його, перевірте branch/version.
setup script перевіряє GPU/nativeBF16, встановлює dataset/tokenizer/plot dependencies
і виконує CUDA matmul+backward. Torch не перевстановлюється. Повторний запуск
pip із pinned versions не створює окремий service. Jupyter забезпечує template.

Перевірте, що /workspace реально знаходиться напостійному mount:

```bash
findmnt -T /workspace
nvidia-smi
```

Якщо findmnt відсутній, подивіться mount у Pod configuration/`df -h /workspace`.
Саме існування каталогу /workspace не гарантує persistence.

## 4. Upload та extraction

У Jupyter file browser /workspace завантажте next-moe-runpod-transfer.zip.
Для великих ZIP можна застосувати SFTP/інший підтриманий RunPod upload спосіб;
Jupyter має бути доступний через Connect. Звичайний Google Drive mount не працює
як уColab. Не переносимо credentials або Colab token files.

Terminal:

```bash
python - <<'PY'
from pathlib import Path
import zipfile
archive=Path('/workspace/next-moe-runpod-transfer.zip')
output=Path('/workspace/next-moe-transfer')
output.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
    for member in z.infolist():
        p=Path(member.filename)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('Unsafe archive path')
    z.extractall(output)
print(output)
PY
```

У /workspace/next-moe-transfer мають лежати weights.pt, data-replay/ та optional
phase.json,last.pt, а не додаткова вкладена папка. Не повторювати extraction
поверх існуючого input. ZIP можна прибрати після перевірки, залишивши зовнішню копію.

## 5. Відкрити RunPod notebook

Відкрити `/workspace/next-moe/notebooks/next_moe_runpod.ipynb` уJupyter.
Код автономний, не потребує Git imports repository длясамоїмоделі.
INPUT_PATH=/workspace/next-moe-transfer, STORAGE_PATH=/workspace/next-moe-runs.
Default RUN_NAME=runpod-replay-v1, ADDITIONAL_TOKENS=0, SESSION_UPDATES=None.

**ADDITIONAL_TOKENS=0 переносить залишок з phase.json**, не додає ще10M.
Якщо старий budget завершений або phase.json немає, він не почне нове навчання.
Для явно нового+10M: ADDITIONAL_TOKENS=10_000_000 та **новий RUN_NAME**.
Для уже незавершеного+10M залишити0, щоб не продовжити target помилково.

Auto: власний volume last.pt → власний weights.pt → imported full/compact.
Full initial exact migration лише при ADDITIONAL_TOKENS=0; нова фаза explicitwarm.
Перевірка source hashes, token signature та exact recipe. NativeBF16 має
підтримуватися для oldBF16 exact migration; інша precision потребує warm mode.
Для warm LR5e-5,ramp100,token-cosine з цільовогоphase. Якщо input phase немає,
задайте новий бюджет, не вигадуйте historical target.

Встановлюються replay sampler та old/new domain evaluation. НовийGPU не змінює
архітектуру. Первиннаvalidation — перевірка ваг; невелика числова різниця можлива
через precision/hardware. Завантаження корпуса не повторюється.

Для незавершеноїфази benchmark після одного реальногоAdam allocation update
порівнюєmicrobatch1/2/4 з однаковим2048tokens/update. Тестовийprogress відкочується.
PRICE_PER_HOUR_USD=.24 змініть підвашPod. tokens/$=speed*3600/price.
За завершеногоbudget benchmark пропускається, не створюєнеузгодженийreal update.
Першуtrainingcell можнаобмежити SESSION_UPDATES=20 дляsmoke; після перевірки
last.pt і відновлення збільшити доNone длязалишку. Дляграфіківпоставити500.

## 6. Checkpoint і завершенняPod

Наpersistent volume: last.pt≈3.83GiB (full), weights.pt≈0.55GiB, data-replay/,
phase.json, configs, source_hashes, logs/pip-freeze. Повний last.pt atomic-save
потребуєщеприблизно3.83GiB тимчасовомісця. Notebook перевіряє9GiB freeнаvolume.
Save кожні500updates івкінці. ПриStop trainingcell коднамагаєтьсяsave;
раптовеTerminate/OOM процесу може залишити лише попередній completedcheckpoint.
ЗалиштеbackupпозаPod/networkvolume передвидаленням. Snapshot/hash не гарантує
сховище при видаленні. IdlePod такожспоживаєGPUоплату; Stopможезберігатиstoragebilling.

## Що перевірено тут

CPUtests: compact/fullZIP, SHA manifest, незміннийcorpus, checkpoint/Adam/RNG
migration, fixed imported target, навчання наtiny, graphs/generation, exactreopen,
відмова за corruptedtransfer. shell script syntax перевірений. Нашасередовищене
маєCUDAGPU; справжнійsetupCUDA/Jupyter/volume іspeed потребують запускунаRunPod.
