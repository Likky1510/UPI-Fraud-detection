const API_BASE = "http://127.0.0.1:8000";
const LANGUAGE_LABELS = { en: "English", hi: "Hindi", te: "Telugu" };
const UI_TEXT = {
  en: {
    subtitle: "Fraud Detection Console",
    navDashboard: "Dashboard",
    navHome: "Home",
    navLogout: "Logout",
    languageLabel: "Alert Language",
    mainTitle: "UPI Fraud Detection & Awareness",
    cardTotal: "Total",
    cardSafe: "Safe",
    cardRisky: "Risky",
    cardBlocked: "Blocked",
    realTimeTitle: "Real-Time Transaction Check",
    transaction_id: "Transaction ID",
    amount: "Amount",
    step: "Step",
    oldbalanceOrg: "Old Balance Org",
    newbalanceOrig: "New Balance Org",
    oldbalanceDest: "Old Balance Dest",
    newbalanceDest: "New Balance Dest",
    recent_txn_count_1h: "Txns Last 1h",
    typePayment: "PAYMENT",
    typeTransfer: "TRANSFER",
    typeCashOut: "CASH_OUT",
    typeCashIn: "CASH_IN",
    typeDebit: "DEBIT",
    deviceChangedLabel: "Device Changed",
    locationChangedLabel: "Location Changed",
    scoreBtn: "Score Transaction",
    batchTitle: "Batch Simulation",
    simulateBtn: "Run Simulation",
    recentTitle: "Recent Scored Transactions",
    thId: "ID",
    thType: "Type",
    thAmount: "Amount",
    thVerdict: "Verdict",
    thRisk: "Risk",
    thTip: "Tip",
    chartSafe: "Safe",
    chartRisky: "Risky",
    chartBlocked: "Blocked",
    checkingBackend: "Checking backend...",
    backendOnlineModel: "Backend online | Model loaded",
    backendOnlineHeuristic: "Backend online | Heuristic mode",
    backendOffline: "Backend offline. Start FastAPI on port 8000.",
    defaultResult: "Submit a transaction to see risk score and safety tips.",
    languageChanged: "Language changed to {lang}. Score a transaction to see localized tips.",
    unableScore: "Unable to score transaction. Verify backend is running and try again.",
    simulating: "Simulating {count} transactions in {lang}...",
    simComplete: "Simulation complete: {total} processed | Blocked: {blocked} | Risky: {risky} | Language: {lang}",
    simFailed: "Simulation failed. Check backend logs and retry.",
    resultFormat: "{verdict} | Risk: {risk}% | Language: {lang} | {tip}",
    voiceTitle: "Voice Assistant",
    voiceDesc: "Get spoken guidance to fill the form correctly and avoid UPI fraud.",
    guideSpeakBtn: "Speak Form Guide",
    tipsSpeakBtn: "Speak Safety Tips",
    voiceListenBtn: "Start Voice Input",
    voiceStopBtn: "Stop Voice",
    assistantQuery: "Type a voice command like: help, safe example, tips",
    assistantAskBtn: "Ask Assistant",
    autoVoiceLabel: "Speak assistant replies automatically",
    voiceReady: "Voice assistant ready.",
    voiceNotSupported: "Voice input is not supported in this browser. Use typed assistant command.",
    voiceListening: "Listening... say help, tips, safe example, risky example, score, or simulate.",
    voiceHeard: "Heard: {text}",
    voiceCommandUnknown: "Command not recognized. Try: help, tips, safe example, risky example, score, simulate.",
    voiceStopped: "Voice assistant stopped.",
    voiceGuide: "To fill the form correctly: enter transaction id, choose transaction type, enter amount, step from 1 to 744, sender old and new balances, receiver old and new balances, recent one hour transaction count, and mark device or location changed if unusual.",
    voiceTips: "Never share OTP or UPI PIN. Verify UPI ID and recipient name before payment. Avoid unknown QR codes, urgent refund links, and fake customer care calls.",
    voiceFilledSafe: "Filled a safe example. You can now click score transaction.",
    voiceFilledRisky: "Filled a risky example. You can now click score transaction.",
    voiceScoring: "Scoring transaction now.",
    voiceSimulating: "Starting batch simulation now.",
  },
  hi: {
    subtitle: "धोखाधड़ी पहचान कंसोल",
    navDashboard: "डैशबोर्ड",
    navHome: "होम",
    navLogout: "लॉगआउट",
    languageLabel: "अलर्ट भाषा",
    mainTitle: "UPI धोखाधड़ी पहचान और जागरूकता",
    cardTotal: "कुल",
    cardSafe: "सुरक्षित",
    cardRisky: "जोखिमपूर्ण",
    cardBlocked: "ब्लॉक",
    realTimeTitle: "रियल-टाइम लेनदेन जांच",
    transaction_id: "लेनदेन आईडी",
    amount: "राशि",
    step: "स्टेप",
    oldbalanceOrg: "पुराना शेष (प्रेषक)",
    newbalanceOrig: "नया शेष (प्रेषक)",
    oldbalanceDest: "पुराना शेष (प्राप्तकर्ता)",
    newbalanceDest: "नया शेष (प्राप्तकर्ता)",
    recent_txn_count_1h: "पिछले 1 घंटे के लेनदेन",
    typePayment: "भुगतान (PAYMENT)",
    typeTransfer: "ट्रांसफर (TRANSFER)",
    typeCashOut: "कैश आउट (CASH_OUT)",
    typeCashIn: "कैश इन (CASH_IN)",
    typeDebit: "डेबिट (DEBIT)",
    deviceChangedLabel: "डिवाइस बदला",
    locationChangedLabel: "लोकेशन बदली",
    scoreBtn: "लेनदेन स्कोर करें",
    batchTitle: "बैच सिमुलेशन",
    simulateBtn: "सिमुलेशन चलाएं",
    recentTitle: "हाल के स्कोर किए गए लेनदेन",
    thId: "आईडी",
    thType: "प्रकार",
    thAmount: "राशि",
    thVerdict: "निर्णय",
    thRisk: "जोखिम",
    thTip: "सुझाव",
    chartSafe: "सुरक्षित",
    chartRisky: "जोखिमपूर्ण",
    chartBlocked: "ब्लॉक",
    checkingBackend: "बैकएंड जांच हो रही है...",
    backendOnlineModel: "बैकएंड चालू | मॉडल लोडेड",
    backendOnlineHeuristic: "बैकएंड चालू | हीयूरिस्टिक मोड",
    backendOffline: "बैकएंड ऑफलाइन है। FastAPI को पोर्ट 8000 पर शुरू करें।",
    defaultResult: "जोखिम स्कोर और सुरक्षा सुझाव देखने के लिए लेनदेन सबमिट करें।",
    languageChanged: "भाषा {lang} कर दी गई है। स्थानीय सुझाव देखने के लिए लेनदेन स्कोर करें।",
    unableScore: "लेनदेन स्कोर नहीं हुआ। बैकएंड चल रहा है या नहीं जांचें और फिर कोशिश करें।",
    simulating: "{lang} में {count} लेनदेन का सिमुलेशन चल रहा है...",
    simComplete: "सिमुलेशन पूरा: {total} प्रोसेस्ड | ब्लॉक: {blocked} | जोखिमपूर्ण: {risky} | भाषा: {lang}",
    simFailed: "सिमुलेशन विफल हुआ। बैकएंड लॉग जांचें और फिर कोशिश करें।",
    resultFormat: "{verdict} | जोखिम: {risk}% | भाषा: {lang} | {tip}",
    voiceTitle: "वॉइस असिस्टेंट",
    voiceDesc: "फॉर्म सही भरने और UPI फ्रॉड से बचने के लिए आवाज़ में मार्गदर्शन लें।",
    guideSpeakBtn: "फॉर्म गाइड सुनें",
    tipsSpeakBtn: "सुरक्षा टिप्स सुनें",
    voiceListenBtn: "वॉइस इनपुट शुरू करें",
    voiceStopBtn: "वॉइस बंद करें",
    assistantQuery: "कमांड लिखें: help, safe example, tips",
    assistantAskBtn: "असिस्टेंट पूछें",
    autoVoiceLabel: "असिस्टेंट जवाब अपने आप बोलें",
    voiceReady: "वॉइस असिस्टेंट तैयार है।",
    voiceNotSupported: "इस ब्राउज़र में वॉइस इनपुट समर्थित नहीं है। टाइप करके कमांड दें।",
    voiceListening: "सुन रहा है... बोलें: help, tips, safe example, risky example, score, simulate.",
    voiceHeard: "सुना गया: {text}",
    voiceCommandUnknown: "कमांड समझ नहीं आया। कोशिश करें: help, tips, safe example, risky example, score, simulate.",
    voiceStopped: "वॉइस असिस्टेंट बंद किया गया।",
    voiceGuide: "फॉर्म भरने के लिए: ट्रांजैक्शन आईडी डालें, प्रकार चुनें, राशि डालें, स्टेप 1 से 744 तक रखें, प्रेषक और प्राप्तकर्ता के पुराने और नए बैलेंस भरें, पिछले 1 घंटे के लेनदेन भरें, और जरूरत हो तो डिवाइस या लोकेशन बदली चुनें।",
    voiceTips: "OTP या UPI PIN कभी साझा न करें। भुगतान से पहले UPI आईडी और नाम जांचें। अज्ञात QR कोड, तुरंत रिफंड लिंक और फर्जी कस्टमर केयर कॉल से बचें।",
    voiceFilledSafe: "सुरक्षित उदाहरण भर दिया गया है। अब आप स्कोर ट्रांजैक्शन क्लिक करें।",
    voiceFilledRisky: "जोखिमपूर्ण उदाहरण भर दिया गया है। अब आप स्कोर ट्रांजैक्शन क्लिक करें।",
    voiceScoring: "अब ट्रांजैक्शन स्कोर किया जा रहा है।",
    voiceSimulating: "अब बैच सिमुलेशन शुरू किया जा रहा है।",
  },
  te: {
    subtitle: "మోసం గుర్తింపు కన్సోల్",
    navDashboard: "డాష్‌బోర్డ్",
    navHome: "హోమ్",
    navLogout: "లాగ్ అవుట్",
    languageLabel: "అలర్ట్ భాష",
    mainTitle: "UPI మోసం గుర్తింపు మరియు అవగాహన",
    cardTotal: "మొత్తం",
    cardSafe: "సురక్షితం",
    cardRisky: "ప్రమాదకరం",
    cardBlocked: "నిలిపివేసినవి",
    realTimeTitle: "రియల్-టైమ్ లావాదేవీ తనిఖీ",
    transaction_id: "లావాదేవీ ఐడి",
    amount: "మొత్తం",
    step: "స్టెప్",
    oldbalanceOrg: "పాత బ్యాలెన్స్ (సెండర్)",
    newbalanceOrig: "కొత్త బ్యాలెన్స్ (సెండర్)",
    oldbalanceDest: "పాత బ్యాలెన్స్ (రిసీవర్)",
    newbalanceDest: "కొత్త బ్యాలెన్స్ (రిసీవర్)",
    recent_txn_count_1h: "గత 1 గంట లావాదేవీలు",
    typePayment: "చెల్లింపు (PAYMENT)",
    typeTransfer: "బదిలీ (TRANSFER)",
    typeCashOut: "క్యాష్ అవుట్ (CASH_OUT)",
    typeCashIn: "క్యాష్ ఇన్ (CASH_IN)",
    typeDebit: "డెబిట్ (DEBIT)",
    deviceChangedLabel: "డివైస్ మారింది",
    locationChangedLabel: "లోకేషన్ మారింది",
    scoreBtn: "లావాదేవీ స్కోర్ చేయండి",
    batchTitle: "బ్యాచ్ సిమ్యులేషన్",
    simulateBtn: "సిమ్యులేషన్ నడపండి",
    recentTitle: "ఇటీవల స్కోర్ చేసిన లావాదేవీలు",
    thId: "ఐడి",
    thType: "రకం",
    thAmount: "మొత్తం",
    thVerdict: "తీర్పు",
    thRisk: "ప్రమాదం",
    thTip: "సలహా",
    chartSafe: "సురక్షితం",
    chartRisky: "ప్రమాదకరం",
    chartBlocked: "నిలిపివేసినవి",
    checkingBackend: "బ్యాకెండ్‌ను తనిఖీ చేస్తోంది...",
    backendOnlineModel: "బ్యాకెండ్ ఆన్‌లైన్ | మోడల్ లోడ్ అయింది",
    backendOnlineHeuristic: "బ్యాకెండ్ ఆన్‌లైన్ | హ్యూరిస్టిక్ మోడ్",
    backendOffline: "బ్యాకెండ్ ఆఫ్‌లైన్‌లో ఉంది. FastAPIని పోర్ట్ 8000లో ప్రారంభించండి.",
    defaultResult: "రిస్క్ స్కోర్ మరియు భద్రతా సూచనలు చూడడానికి లావాదేవీని సమర్పించండి.",
    languageChanged: "భాషను {lang}గా మార్చాం. స్థానిక సూచనలు చూడటానికి లావాదేవీని స్కోర్ చేయండి.",
    unableScore: "లావాదేవీ స్కోర్ కాలేదు. బ్యాకెండ్ నడుస్తోందో లేదో చూసి మళ్లీ ప్రయత్నించండి.",
    simulating: "{lang}లో {count} లావాదేవీల సిమ్యులేషన్ జరుగుతోంది...",
    simComplete: "సిమ్యులేషన్ పూర్తి: {total} ప్రాసెస్ అయ్యాయి | నిలిపివేసినవి: {blocked} | ప్రమాదకరం: {risky} | భాష: {lang}",
    simFailed: "సిమ్యులేషన్ విఫలమైంది. బ్యాకెండ్ లాగ్స్ చూడండి మరియు మళ్లీ ప్రయత్నించండి.",
    resultFormat: "{verdict} | ప్రమాదం: {risk}% | భాష: {lang} | {tip}",
    voiceTitle: "వాయిస్ అసిస్టెంట్",
    voiceDesc: "ఫారమ్ సరైన విధంగా నింపడం మరియు UPI మోసం నివారణకు వాయిస్ మార్గదర్శనం పొందండి.",
    guideSpeakBtn: "ఫారమ్ గైడ్ వినండి",
    tipsSpeakBtn: "భద్రతా సూచనలు వినండి",
    voiceListenBtn: "వాయిస్ ఇన్‌పుట్ ప్రారంభించండి",
    voiceStopBtn: "వాయిస్ ఆపండి",
    assistantQuery: "కమాండ్ టైప్ చేయండి: help, safe example, tips",
    assistantAskBtn: "అసిస్టెంట్‌ను అడగండి",
    autoVoiceLabel: "అసిస్టెంట్ సమాధానాలను ఆటోమేటిక్‌గా చదవండి",
    voiceReady: "వాయిస్ అసిస్టెంట్ సిద్ధంగా ఉంది.",
    voiceNotSupported: "ఈ బ్రౌజర్‌లో వాయిస్ ఇన్‌పుట్ లేదు. టైప్ చేసి కమాండ్ ఇవ్వండి.",
    voiceListening: "వింటోంది... చెప్పండి: help, tips, safe example, risky example, score, simulate.",
    voiceHeard: "విన్నది: {text}",
    voiceCommandUnknown: "కమాండ్ అర్థం కాలేదు. ప్రయత్నించండి: help, tips, safe example, risky example, score, simulate.",
    voiceStopped: "వాయిస్ అసిస్టెంట్ ఆపబడింది.",
    voiceGuide: "ఫారమ్ నింపడానికి: ట్రాన్సాక్షన్ ఐడి ఇవ్వండి, రకం ఎంచుకోండి, మొత్తం ఇవ్వండి, స్టెప్ 1 నుండి 744 మధ్య ఇవ్వండి, పంపినవారి మరియు స్వీకరించినవారి పాత మరియు కొత్త బ్యాలెన్స్‌లు ఇవ్వండి, గత 1 గంట లావాదేవీల సంఖ్య ఇవ్వండి, అవసరమైతే డివైస్ లేదా లోకేషన్ మార్పు ఎంపిక చేయండి.",
    voiceTips: "OTP లేదా UPI PIN ఎప్పుడూ పంచుకోవద్దు. చెల్లింపుకు ముందు UPI ID మరియు పేరు ధృవీకరించండి. తెలియని QR కోడ్‌లు, త్వరిత రిఫండ్ లింకులు మరియు నకిలీ కస్టమర్ కేర్ కాల్స్ నుండి జాగ్రత్తగా ఉండండి.",
    voiceFilledSafe: "సురక్షిత ఉదాహరణను నింపాం. ఇప్పుడు స్కోర్ ట్రాన్సాక్షన్ నొక్కండి.",
    voiceFilledRisky: "ప్రమాదకర ఉదాహరణను నింపాం. ఇప్పుడు స్కోర్ ట్రాన్సాక్షన్ నొక్కండి.",
    voiceScoring: "ఇప్పుడు ట్రాన్సాక్షన్ స్కోర్ చేస్తోంది.",
    voiceSimulating: "ఇప్పుడు బ్యాచ్ సిమ్యులేషన్ ప్రారంభమవుతోంది.",
  },
};

const refs = {
  apiState: document.getElementById("apiState"),
  language: document.getElementById("language"),
  total: document.getElementById("total"),
  safe: document.getElementById("safe"),
  risky: document.getElementById("risky"),
  blocked: document.getElementById("blocked"),
  table: document.getElementById("tableData"),
  result: document.getElementById("latestResult"),
  form: document.getElementById("txnForm"),
  simulateBtn: document.getElementById("simulateBtn"),
  batchCount: document.getElementById("batchCount"),
  guideSpeakBtn: document.getElementById("guideSpeakBtn"),
  tipsSpeakBtn: document.getElementById("tipsSpeakBtn"),
  voiceListenBtn: document.getElementById("voiceListenBtn"),
  voiceStopBtn: document.getElementById("voiceStopBtn"),
  assistantQuery: document.getElementById("assistantQuery"),
  assistantAskBtn: document.getElementById("assistantAskBtn"),
  voiceStatus: document.getElementById("voiceStatus"),
  autoVoiceTips: document.getElementById("autoVoiceTips"),
  assistantMessages: document.getElementById("assistantMessages"),
};

let summary = { total: 0, safe: 0, risky: 0, blocked: 0 };
let chart;
let currentLang = "en";
let recognition = null;
let isListening = false;
const SPEECH_LANG = { en: "en-IN", hi: "hi-IN", te: "te-IN" };
const ASSISTANT_KB = {
  en: {
    welcome: "I am your UPI Shield assistant. Ask me anything about this dashboard, form fields, scoring, simulation, or fraud safety.",
    fallback: "I can help with: how to fill form, step, amount/balances, safe vs risky vs blocked, simulation, language change, and fraud tips.",
    guide: "Fill in this order: Transaction ID, Type, Amount, Step (1-744), sender balances, receiver balances, recent txns in 1 hour, then device/location changed if unusual.",
    tips: "Never share OTP or UPI PIN. Verify UPI ID and recipient name. Avoid unknown QR codes and fake support calls.",
    step: "Step is the dataset time index in hours. Use any integer from 1 to 744.",
    amount: "Amount is the transfer value. Keep balances consistent: sender new balance is usually old minus amount.",
    balances: "Old Balance Org/Dest means before transaction. New Balance Org/Dest means after transaction.",
    device: "Enable Device Changed only if the transaction is from a new or unusual device.",
    location: "Enable Location Changed only if the payment location is unusual for the user.",
    txnCount: "Txns Last 1h means number of recent transactions by same user in the last hour.",
    cashin: "CASH_IN means money coming into account. CASH_OUT means money going out.",
    verdicts: "SAFE = low risk, RISKY = suspicious, BLOCKED = high risk and blocked by policy.",
    simulation: "Batch Simulation runs large synthetic transactions and gives Safe/Risky/Blocked counts.",
    language: "Use Alert Language on the left to switch English, Hindi, Telugu for full UI and tips.",
    backend: "Backend status is shown at top-right. If offline, start FastAPI on port 8000.",
    scored: "Scoring transaction now.",
    simStart: "Starting simulation now.",
    filledSafe: "Filled a safe example. You can click Score Transaction.",
    filledRisky: "Filled a risky example. You can click Score Transaction.",
    langChanged: "Language changed to {lang}.",
  },
  hi: {
    welcome: "मैं आपका UPI Shield असिस्टेंट हूं। डैशबोर्ड, फॉर्म, स्कोरिंग, सिमुलेशन और सुरक्षा पर कुछ भी पूछें।",
    fallback: "मैं इन विषयों में मदद कर सकता हूं: फॉर्म कैसे भरें, step, amount/balance, safe/risky/blocked, simulation, language change, fraud tips.",
    guide: "यह क्रम रखें: Transaction ID, Type, Amount, Step (1-744), sender balances, receiver balances, पिछले 1 घंटे के txns, और जरूरत हो तो device/location changed.",
    tips: "OTP या UPI PIN कभी साझा न करें। UPI ID और नाम जांचें। अज्ञात QR और फर्जी सपोर्ट कॉल से बचें।",
    step: "Step डेटा का घंटों वाला समय इंडेक्स है। 1 से 744 तक कोई पूर्णांक दें।",
    amount: "Amount ट्रांसफर राशि है। बैलेंस सही रखें: sender new balance आमतौर पर old minus amount होता है।",
    balances: "Old Balance Org/Dest = लेनदेन से पहले। New Balance Org/Dest = लेनदेन के बाद।",
    device: "Device Changed तभी चुनें जब डिवाइस नया या असामान्य हो।",
    location: "Location Changed तभी चुनें जब स्थान असामान्य हो।",
    txnCount: "Txns Last 1h मतलब पिछले 1 घंटे में उसी यूजर के लेनदेन की संख्या।",
    cashin: "CASH_IN मतलब पैसे खाते में आ रहे हैं। CASH_OUT मतलब पैसे बाहर जा रहे हैं।",
    verdicts: "SAFE = कम जोखिम, RISKY = संदिग्ध, BLOCKED = उच्च जोखिम और ब्लॉक।",
    simulation: "Batch Simulation बड़े synthetic लेनदेन चलाकर Safe/Risky/Blocked काउंट देता है।",
    language: "बाईं तरफ Alert Language से English/Hindi/Telugu में पूरा UI बदलिए।",
    backend: "ऊपर दाईं ओर backend status दिखता है। Offline हो तो FastAPI को port 8000 पर चलाएं।",
    scored: "लेनदेन स्कोर किया जा रहा है।",
    simStart: "सिमुलेशन शुरू किया जा रहा है।",
    filledSafe: "सुरक्षित उदाहरण भर दिया गया। अब Score Transaction दबाएं।",
    filledRisky: "जोखिमपूर्ण उदाहरण भर दिया गया। अब Score Transaction दबाएं।",
    langChanged: "भाषा {lang} में बदल दी गई है।",
  },
  te: {
    welcome: "నేను మీ UPI Shield అసిస్టెంట్‌ను. డాష్‌బోర్డ్, ఫారమ్, స్కోరింగ్, సిమ్యులేషన్, భద్రతపై ఏదైనా అడగండి.",
    fallback: "నేను ఈ విషయాల్లో సహాయం చేస్తాను: ఫారమ్ ఎలా నింపాలి, step, amount/balance, safe/risky/blocked, simulation, language change, fraud tips.",
    guide: "ఈ క్రమంలో నింపండి: Transaction ID, Type, Amount, Step (1-744), sender balances, receiver balances, గత 1 గంట txn count, అవసరమైతే device/location changed.",
    tips: "OTP లేదా UPI PIN ఎప్పుడూ పంచుకోవద్దు. UPI ID మరియు పేరు ధృవీకరించండి. తెలియని QR కోడ్‌లు, నకిలీ సపోర్ట్ కాల్స్ నుండి జాగ్రత్త.",
    step: "Step అనేది గంటల ఆధారిత టైమ్ ఇండెక్స్. 1 నుండి 744 వరకు పూర్తి సంఖ్య ఇవ్వండి.",
    amount: "Amount అంటే పంపే మొత్తం. బ్యాలెన్స్ సరిపోయేలా ఉండాలి: sender new balance సాధారణంగా old minus amount.",
    balances: "Old Balance Org/Dest = లావాదేవీకి ముందు. New Balance Org/Dest = లావాదేవీ తర్వాత.",
    device: "Device Changed ను కొత్త/అసాధారణ డివైస్ అయితే మాత్రమే ఎంచుకోండి.",
    location: "Location Changed ను అసాధారణ ప్రాంతం అయితే మాత్రమే ఎంచుకోండి.",
    txnCount: "Txns Last 1h అంటే అదే యూజర్ గత 1 గంటలో చేసిన లావాదేవీల సంఖ్య.",
    cashin: "CASH_IN అంటే డబ్బు ఖాతాలోకి రావడం. CASH_OUT అంటే డబ్బు బయటకు వెళ్లడం.",
    verdicts: "SAFE = తక్కువ ప్రమాదం, RISKY = అనుమానాస్పదం, BLOCKED = అధిక ప్రమాదం మరియు నిలిపివేత.",
    simulation: "Batch Simulation పెద్ద synthetic లావాదేవీలు నడిపి Safe/Risky/Blocked కౌంట్ చూపిస్తుంది.",
    language: "ఎడమవైపు Alert Language ద్వారా English/Hindi/Telugu కి పూర్తి UI మార్చండి.",
    backend: "పై కుడివైపు backend status కనిపిస్తుంది. Offline అయితే FastAPIని port 8000లో ప్రారంభించండి.",
    scored: "ఇప్పుడు లావాదేవీ స్కోర్ అవుతోంది.",
    simStart: "ఇప్పుడు సిమ్యులేషన్ ప్రారంభమవుతోంది.",
    filledSafe: "సురక్షిత ఉదాహరణను నింపాం. ఇప్పుడు Score Transaction నొక్కండి.",
    filledRisky: "ప్రమాదకర ఉదాహరణను నింపాం. ఇప్పుడు Score Transaction నొక్కండి.",
    langChanged: "భాషను {lang}కి మార్చాం.",
  },
};

function t(key) {
  const bucket = UI_TEXT[currentLang] || UI_TEXT.en;
  return bucket[key] || UI_TEXT.en[key] || key;
}

function formatMessage(template, vars) {
  return Object.entries(vars).reduce((acc, [k, v]) => acc.replaceAll(`{${k}}`, String(v)), template);
}

function setVoiceStatus(text) {
  refs.voiceStatus.textContent = text;
}

function kb(key) {
  const bucket = ASSISTANT_KB[currentLang] || ASSISTANT_KB.en;
  return bucket[key] || ASSISTANT_KB.en[key] || "";
}

function addChatMessage(text, role = "bot") {
  const node = document.createElement("div");
  node.className = role === "user" ? "msg msg-user" : "msg msg-bot";
  node.textContent = text;
  refs.assistantMessages.appendChild(node);
  refs.assistantMessages.scrollTop = refs.assistantMessages.scrollHeight;
}

function speakText(text) {
  if (!("speechSynthesis" in window)) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = SPEECH_LANG[currentLang] || "en-IN";
  const voices = window.speechSynthesis.getVoices();
  const preferredVoice = voices.find((v) => v.lang.toLowerCase().startsWith((SPEECH_LANG[currentLang] || "en").slice(0, 2)));
  if (preferredVoice) utterance.voice = preferredVoice;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

function fillSafeExample() {
  document.getElementById("transaction_id").value = `TXN-SAFE-${Date.now().toString().slice(-5)}`;
  document.getElementById("type").value = "PAYMENT";
  document.getElementById("amount").value = "500";
  document.getElementById("step").value = "120";
  document.getElementById("oldbalanceOrg").value = "10000";
  document.getElementById("newbalanceOrig").value = "9500";
  document.getElementById("oldbalanceDest").value = "22000";
  document.getElementById("newbalanceDest").value = "22500";
  document.getElementById("recent_txn_count_1h").value = "1";
  document.getElementById("device_changed").checked = false;
  document.getElementById("location_changed").checked = false;
}

function fillRiskyExample() {
  document.getElementById("transaction_id").value = `TXN-RISK-${Date.now().toString().slice(-5)}`;
  document.getElementById("type").value = "TRANSFER";
  document.getElementById("amount").value = "240000";
  document.getElementById("step").value = "700";
  document.getElementById("oldbalanceOrg").value = "245000";
  document.getElementById("newbalanceOrig").value = "0";
  document.getElementById("oldbalanceDest").value = "500";
  document.getElementById("newbalanceDest").value = "240500";
  document.getElementById("recent_txn_count_1h").value = "8";
  document.getElementById("device_changed").checked = true;
  document.getElementById("location_changed").checked = true;
}

function commandMatches(command, patterns) {
  return patterns.some((pattern) => command.includes(pattern));
}

function runAssistantCommand(rawCommand) {
  const command = (rawCommand || "").trim().toLowerCase();
  if (!command) return;
  addChatMessage(rawCommand, "user");

  const helpPatterns = ["help", "guide", "how to fill", "मदद", "గైడ్", "సహాయం", "fill form", "form"];
  const tipsPatterns = ["tips", "safety", "fraud", "otp", "सुरक्षा", "సూచనలు", "భద్రత"];
  const safePatterns = ["safe example", "safe form", "सुरक्षित उदाहरण", "సురక్షిత ఉదాహరణ"];
  const riskyPatterns = ["risky example", "fraud example", "जोखिम", "ప్రమాద"];
  const scorePatterns = ["score", "submit", "check transaction", "स्कोर", "స్కోర్"];
  const simulatePatterns = ["simulate", "batch", "simulation", "सिमुलेशन", "సిమ్యులేషన్"];
  const stepPatterns = ["step", "स्टेप", "స్టెప్"];
  const amountPatterns = ["amount", "राशि", "మొత్తం"];
  const balancePatterns = ["balance", "old balance", "new balance", "बैलेंस", "బ్యాలెన్స్"];
  const devicePatterns = ["device changed", "device", "डिवाइस", "డివైస్"];
  const locationPatterns = ["location changed", "location", "लोकेशन", "లోకేషన్"];
  const txnCountPatterns = ["txns", "last 1h", "1 hour", "1 घंटे", "1 గంట"];
  const cashinPatterns = ["cash in", "cash-out", "cash out", "cashi", "CASH_IN", "CASH_OUT"];
  const verdictPatterns = ["safe risky blocked", "verdict", "blocked", "risky", "safe"];
  const backendPatterns = ["backend", "server", "health", "api"];
  const languagePatterns = ["language", "hindi", "english", "telugu", "भाषा", "భాష"];
  const clearPatterns = ["clear", "reset", "रीसेट", "రిసెట్"];
  const stopPatterns = ["stop", "cancel voice", "వాయిస్ ఆపు", "वॉइस बंद"];

  let response = "";

  if (commandMatches(command, languagePatterns) && (command.includes("change") || command.includes("switch") || command.includes("to "))) {
    if (command.includes("hindi") || command.includes("hi")) refs.language.value = "hi";
    else if (command.includes("telugu") || command.includes("te")) refs.language.value = "te";
    else if (command.includes("english") || command.includes("en")) refs.language.value = "en";
    applyLanguage(refs.language.value);
    response = formatMessage(kb("langChanged"), { lang: LANGUAGE_LABELS[refs.language.value] });
  } else if (commandMatches(command, helpPatterns)) {
    response = kb("guide");
  } else if (commandMatches(command, tipsPatterns)) {
    response = kb("tips");
  } else if (commandMatches(command, stepPatterns)) {
    response = kb("step");
  } else if (commandMatches(command, amountPatterns)) {
    response = kb("amount");
  } else if (commandMatches(command, balancePatterns)) {
    response = kb("balances");
  } else if (commandMatches(command, devicePatterns)) {
    response = kb("device");
  } else if (commandMatches(command, locationPatterns)) {
    response = kb("location");
  } else if (commandMatches(command, txnCountPatterns)) {
    response = kb("txnCount");
  } else if (commandMatches(command, cashinPatterns)) {
    response = kb("cashin");
  } else if (commandMatches(command, verdictPatterns)) {
    response = kb("verdicts");
  } else if (commandMatches(command, backendPatterns)) {
    response = kb("backend");
  } else if (commandMatches(command, simulatePatterns)) {
    response = kb("simStart");
    refs.simulateBtn.click();
  } else if (commandMatches(command, scorePatterns)) {
    response = kb("scored");
    refs.form.requestSubmit();
  } else if (commandMatches(command, safePatterns)) {
    fillSafeExample();
    response = kb("filledSafe");
  } else if (commandMatches(command, riskyPatterns)) {
    fillRiskyExample();
    response = kb("filledRisky");
  } else if (commandMatches(command, clearPatterns)) {
    refs.form.reset();
    document.getElementById("recent_txn_count_1h").value = "0";
    response = kb("guide");
  } else if (commandMatches(command, stopPatterns)) {
    stopVoiceAssistant();
    return;
  } else if (command.includes("what") || command.includes("website") || command.includes("dashboard")) {
    response = kb("welcome");
  } else {
    response = kb("fallback");
  }

  addChatMessage(response, "bot");
  setVoiceStatus(response);
  if (refs.autoVoiceTips.checked) speakText(response);
}

function startVoiceAssistant() {
  const RecognitionClass = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!RecognitionClass) {
    setVoiceStatus(t("voiceNotSupported"));
    speakText(t("voiceNotSupported"));
    return;
  }

  if (recognition && isListening) {
    recognition.stop();
  }

  recognition = new RecognitionClass();
  recognition.lang = SPEECH_LANG[currentLang] || "en-IN";
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    isListening = true;
    setVoiceStatus(t("voiceListening"));
  };

  recognition.onresult = (event) => {
    const transcript = event.results?.[0]?.[0]?.transcript || "";
    setVoiceStatus(formatMessage(t("voiceHeard"), { text: transcript }));
    runAssistantCommand(transcript);
  };

  recognition.onerror = () => {
    isListening = false;
    setVoiceStatus(t("voiceCommandUnknown"));
  };

  recognition.onend = () => {
    isListening = false;
  };

  recognition.start();
}

function stopVoiceAssistant() {
  if (recognition && isListening) recognition.stop();
  isListening = false;
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  setVoiceStatus(t("voiceStopped"));
}

function statusClass(verdict) {
  if (verdict === "SAFE") return "tag-safe";
  if (verdict === "RISKY") return "tag-risky";
  return "tag-blocked";
}

function updateSummary() {
  refs.total.textContent = summary.total;
  refs.safe.textContent = summary.safe;
  refs.risky.textContent = summary.risky;
  refs.blocked.textContent = summary.blocked;

  if (!chart) {
    chart = new Chart(document.getElementById("chart"), {
      type: "doughnut",
      data: {
        labels: [t("chartSafe"), t("chartRisky"), t("chartBlocked")],
        datasets: [{ data: [0, 0, 0], backgroundColor: ["#0f9d58", "#e19100", "#d93025"] }],
      },
      options: { responsive: true, maintainAspectRatio: false },
    });
  }

  chart.data.labels = [t("chartSafe"), t("chartRisky"), t("chartBlocked")];
  chart.data.datasets[0].data = [summary.safe, summary.risky, summary.blocked];
  chart.update();
}

function addResultRowValues(transactionId, type, amount, verdict, riskScore, tip) {
  const row = document.createElement("tr");
  row.innerHTML = `
    <td>${transactionId}</td>
    <td>${type}</td>
    <td>${Number(amount).toFixed(2)}</td>
    <td class="${statusClass(verdict)}">${verdict}</td>
    <td>${riskScore}%</td>
    <td>${tip}</td>
  `;
  refs.table.prepend(row);

  while (refs.table.children.length > 25) {
    refs.table.removeChild(refs.table.lastChild);
  }
}

function addResultRow(payload, result) {
  addResultRowValues(result.transaction_id, payload.type, payload.amount, result.verdict, result.risk_score, result.tip);
}

function registerVerdict(verdict) {
  summary.total += 1;
  if (verdict === "SAFE") summary.safe += 1;
  if (verdict === "RISKY") summary.risky += 1;
  if (verdict === "BLOCKED") summary.blocked += 1;
  updateSummary();
}

async function checkBackend() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    refs.apiState.textContent = data.model_ready ? t("backendOnlineModel") : t("backendOnlineHeuristic");
  } catch (_err) {
    refs.apiState.textContent = t("backendOffline");
  }
}

function applyLanguage(lang) {
  currentLang = UI_TEXT[lang] ? lang : "en";
  const fields = [
    "subtitle", "navDashboard", "navHome", "navLogout", "languageLabel", "mainTitle",
    "cardTotal", "cardSafe", "cardRisky", "cardBlocked", "realTimeTitle",
    "deviceChangedLabel", "locationChangedLabel", "scoreBtn", "batchTitle",
    "simulateBtn", "recentTitle", "thId", "thType", "thAmount", "thVerdict", "thRisk", "thTip",
    "typePayment", "typeTransfer", "typeCashOut", "typeCashIn", "typeDebit",
    "voiceTitle", "voiceDesc", "guideSpeakBtn", "tipsSpeakBtn", "voiceListenBtn", "voiceStopBtn",
    "assistantAskBtn", "autoVoiceLabel",
  ];
  fields.forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.textContent = t(id);
  });

  const placeholders = [
    "transaction_id", "amount", "step", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "recent_txn_count_1h", "assistantQuery",
  ];
  placeholders.forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.placeholder = t(id);
  });

  refs.language.options[0].text = LANGUAGE_LABELS.en;
  refs.language.options[1].text = LANGUAGE_LABELS.hi;
  refs.language.options[2].text = LANGUAGE_LABELS.te;
}

function formPayload() {
  return {
    transaction_id: document.getElementById("transaction_id").value,
    step: Number(document.getElementById("step").value),
    type: document.getElementById("type").value,
    amount: Number(document.getElementById("amount").value),
    oldbalanceOrg: Number(document.getElementById("oldbalanceOrg").value),
    newbalanceOrig: Number(document.getElementById("newbalanceOrig").value),
    oldbalanceDest: Number(document.getElementById("oldbalanceDest").value),
    newbalanceDest: Number(document.getElementById("newbalanceDest").value),
    language: refs.language.value,
    device_changed: document.getElementById("device_changed").checked,
    location_changed: document.getElementById("location_changed").checked,
    recent_txn_count_1h: Number(document.getElementById("recent_txn_count_1h").value || 0),
  };
}

refs.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = formPayload();

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error("Failed to score transaction");

    const result = await res.json();
    const activeLanguage = LANGUAGE_LABELS[result.language] || result.language;
    refs.result.textContent = formatMessage(t("resultFormat"), {
      verdict: result.verdict,
      risk: result.risk_score,
      lang: activeLanguage,
      tip: result.tip,
    });
    if (refs.autoVoiceTips.checked) {
      speakText(`${result.verdict}. ${result.tip}`);
    }
    addResultRow(payload, result);
    registerVerdict(result.verdict);
  } catch (_err) {
    refs.result.textContent = t("unableScore");
  }
});

refs.simulateBtn.addEventListener("click", async () => {
  const requested = Number(refs.batchCount.value || 1200);
  const count = Math.max(1, Math.min(250000, requested));
  refs.batchCount.value = count;
  refs.result.textContent = formatMessage(t("simulating"), {
    count,
    lang: LANGUAGE_LABELS[refs.language.value] || refs.language.value,
  });

  try {
    const simulateRes = await fetch(`${API_BASE}/simulate?count=${count}&language=${refs.language.value}&chunk_size=5000&preview=25`);
    if (!simulateRes.ok) throw new Error("Simulation failed");
    const batch = await simulateRes.json();

    summary = {
      total: batch.total,
      safe: batch.safe,
      risky: batch.risky,
      blocked: batch.blocked,
    };
    updateSummary();

    refs.table.innerHTML = "";
    batch.preview_rows.forEach((row) => {
      addResultRowValues(row.transaction_id, row.type, row.amount, row.verdict, row.risk_score, row.tip);
    });

    refs.result.textContent = formatMessage(t("simComplete"), {
      total: batch.total,
      blocked: batch.blocked,
      risky: batch.risky,
      lang: LANGUAGE_LABELS[refs.language.value] || refs.language.value,
    });
    if (refs.autoVoiceTips.checked) {
      speakText(refs.result.textContent);
    }
  } catch (_err) {
    refs.result.textContent = t("simFailed");
  }
});

refs.guideSpeakBtn.addEventListener("click", () => {
  runAssistantCommand("help");
});

refs.tipsSpeakBtn.addEventListener("click", () => {
  runAssistantCommand("tips");
});

refs.voiceListenBtn.addEventListener("click", () => {
  startVoiceAssistant();
});

refs.voiceStopBtn.addEventListener("click", () => {
  stopVoiceAssistant();
});

refs.assistantAskBtn.addEventListener("click", () => {
  const command = refs.assistantQuery.value || "";
  runAssistantCommand(command);
});

refs.assistantQuery.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    runAssistantCommand(refs.assistantQuery.value || "");
  }
});

refs.language.addEventListener("change", () => {
  applyLanguage(refs.language.value);
  checkBackend();
  const langText = formatMessage(t("languageChanged"), {
    lang: LANGUAGE_LABELS[refs.language.value] || refs.language.value,
  });
  refs.result.textContent = langText;
  addChatMessage(langText, "bot");
  setVoiceStatus(t("voiceReady"));
  updateSummary();
});

applyLanguage(refs.language.value);
refs.result.textContent = t("defaultResult");
refs.apiState.textContent = t("checkingBackend");
setVoiceStatus(t("voiceReady"));
addChatMessage(kb("welcome"), "bot");
checkBackend();
updateSummary();
