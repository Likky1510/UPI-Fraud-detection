const API_BASE = "http://127.0.0.1:8000";
const AUTH_KEY = "upi_user";

function ensureAuthenticated() {
  const raw = localStorage.getItem(AUTH_KEY);
  if (!raw) {
    window.location.href = "login.html";
    throw new Error("Unauthenticated");
  }
}

ensureAuthenticated();

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
    scenarioTransaction: "Transaction Risk (Full)",
    scenarioQrScam: "QR Code Scam",
    scenarioPhishing: "Phishing Link Fraud",
    scenarioFakePay: "Fake Payment Screenshot",
    scenarioOtpPin: "OTP / UPI PIN Fraud",
    scenarioCollect: "Collect Request Scam",
    scenarioRemote: "Remote App / Screen Share Scam",
    scenarioCaller: "Unknown Caller Pressure Scam",
    scenarioFakeApp: "Fake UPI App Clone",
    applyScenarioBtn: "Apply Fraud Type",
    scenarioHint_default: "Select fraud type to open required fields only.",
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
    scenarioTransaction: "लेनदेन जोखिम (पूर्ण)",
    scenarioQrScam: "QR कोड धोखाधड़ी",
    scenarioPhishing: "फिशिंग लिंक धोखाधड़ी",
    scenarioFakePay: "फर्जी भुगतान स्क्रीनशॉट",
    scenarioOtpPin: "OTP / UPI PIN धोखाधड़ी",
    scenarioCollect: "कलेक्ट रिक्वेस्ट धोखाधड़ी",
    scenarioRemote: "रिमोट ऐप / स्क्रीन शेयर धोखाधड़ी",
    scenarioCaller: "अज्ञात कॉलर दबाव धोखाधड़ी",
    scenarioFakeApp: "फर्जी UPI ऐप क्लोन",
    applyScenarioBtn: "फ्रॉड प्रकार लागू करें",
    scenarioHint_default: "फ्रॉड प्रकार चुनें, सिस्टम केवल जरूरी फील्ड दिखाएगा।",
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
    scenarioTransaction: "లావాదేవీ రిస్క్ (పూర్తి)",
    scenarioQrScam: "QR కోడ్ మోసం",
    scenarioPhishing: "ఫిషింగ్ లింక్ మోసం",
    scenarioFakePay: "నకిలీ చెల్లింపు స్క్రీన్‌షాట్",
    scenarioOtpPin: "OTP / UPI PIN మోసం",
    scenarioCollect: "కలెక్ట్ రిక్వెస్ట్ మోసం",
    scenarioRemote: "రిమోట్ యాప్ / స్క్రీన్ షేర్ మోసం",
    scenarioCaller: "తెలియని కాలర్ ఒత్తిడి మోసం",
    scenarioFakeApp: "నకిలీ UPI యాప్ క్లోన్",
    applyScenarioBtn: "మోసం రకం వర్తింపజేయండి",
    scenarioHint_default: "మోసం రకం ఎంచుకుంటే అవసరమైన ఫీల్డ్‌లే చూపిస్తాం.",
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
  fraudScenario: document.getElementById("fraudScenario"),
  applyScenarioBtn: document.getElementById("applyScenarioBtn"),
  scenarioHint: document.getElementById("scenarioHint"),
  scenarioQuestions: document.getElementById("scenarioQuestions"),
  scenarioQuestionsFakeApp: document.getElementById("scenarioQuestionsFakeApp"),
  fraudSignalsPanel: document.getElementById("fraudSignalsPanel"),
  complaintType: document.getElementById("complaintType"),
  complaintDetails: document.getElementById("complaintDetails"),
  openComplaintPortalBtn: document.getElementById("openComplaintPortalBtn"),
  complaintStatus: document.getElementById("complaintStatus"),
};

let summary = { total: 0, safe: 0, risky: 0, blocked: 0 };
let chart;
let currentLang = "en";
let recognition = null;
let isListening = false;
const SPEECH_LANG = { en: "en-IN", hi: "hi-IN", te: "te-IN" };
let availableVoices = [];
const USE_CLOUD_TTS = true;
let currentAudio = null;
let currentAudioUrl = null;
const FEMALE_VOICE_HINTS = {
  en: ["zira", "aria", "jenny", "susan", "hazel", "samantha", "female", "woman"],
  hi: ["heera", "kalpana", "swara", "female", "woman"],
  te: ["swara", "female", "woman"],
};
const LANGUAGE_NAME_HINTS = {
  en: ["english", "en-"],
  hi: ["hindi", "hi-", "हिंदी"],
  te: ["telugu", "te-", "తెలుగు"],
};
const SPEECH_REWRITE_MAP = {
  hi: [
    [/transaction id/gi, "लेनदेन आईडी"],
    [/type/gi, "प्रकार"],
    [/amount/gi, "राशि"],
    [/step/gi, "स्टेप"],
    [/sender balances?/gi, "प्रेषक शेष"],
    [/receiver balances?/gi, "प्राप्तकर्ता शेष"],
    [/dashboard/gi, "डैशबोर्ड"],
    [/score/gi, "स्कोर"],
    [/blocked/gi, "ब्लॉक"],
    [/risky/gi, "जोखिमपूर्ण"],
    [/safe/gi, "सुरक्षित"],
  ],
  te: [
    [/transaction id/gi, "లావాదేవీ ఐడి"],
    [/type/gi, "రకం"],
    [/amount/gi, "మొత్తం"],
    [/step/gi, "స్టెప్"],
    [/sender balances?/gi, "పంపినవారి బ్యాలెన్స్"],
    [/receiver balances?/gi, "స్వీకరించే వారి బ్యాలెన్స్"],
    [/dashboard/gi, "డాష్‌బోర్డ్"],
    [/score/gi, "స్కోర్"],
    [/blocked/gi, "నిలిపివేసినది"],
    [/risky/gi, "ప్రమాదకరం"],
    [/safe/gi, "సురక్షితం"],
  ],
};
const ASSISTANT_KB = {
  en: {
    welcome: "I am your UPI Sentinel assistant. Ask me anything about this dashboard, form fields, scoring, simulation, or fraud safety.",
    fallback: "I can help with: how to fill form, step, amount/balances, safe vs risky vs blocked, simulation, language change, and fraud tips.",
    guide: "Fill in this order: Transaction ID, Type, Amount, Step (1-744), sender balances, receiver balances, recent txns in 1 hour, then device/location changed if unusual.",
    tips: "Never share OTP or UPI PIN. Verify UPI ID and recipient name. Avoid unknown QR codes and fake support calls.",
    qrScam: "QR scam: scammers ask you to scan a QR to receive money. Receiving money never needs UPI PIN. Do not scan unknown QR codes.",
    phishingScam: "Phishing fraud: fake payment links or fake support pages steal your details. Open only official app links.",
    fakePaymentScam: "Fake payment scam: fraudsters show edited screenshots. Always verify actual credit in your UPI app or bank SMS.",
    collectScam: "Collect request scam: unknown ID sends collect request and asks you to approve. Reject suspicious collect requests.",
    otpPinScam: "OTP/PIN fraud: never share OTP or UPI PIN. Bank staff never ask for them.",
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
    welcome: "मैं आपका UPI Sentinel असिस्टेंट हूं। डैशबोर्ड, फॉर्म, स्कोरिंग, सिमुलेशन और सुरक्षा पर कुछ भी पूछें।",
    fallback: "मैं इन विषयों में मदद कर सकता हूं: फॉर्म कैसे भरें, स्टेप, राशि/बैलेंस, सुरक्षित/जोखिमपूर्ण/ब्लॉक, सिमुलेशन, भाषा बदलना और सुरक्षा सुझाव।",
    guide: "यह क्रम रखें: लेनदेन आईडी, प्रकार, राशि, स्टेप (1-744), प्रेषक बैलेंस, प्राप्तकर्ता बैलेंस, पिछले 1 घंटे के लेनदेन, और जरूरत हो तो डिवाइस/लोकेशन बदला चुनें।",
    tips: "OTP या UPI PIN कभी साझा न करें। UPI ID और नाम जांचें। अज्ञात QR और फर्जी सपोर्ट कॉल से बचें।",
    qrScam: "QR धोखाधड़ी: ठग पैसा प्राप्त करने के नाम पर QR स्कैन करवाते हैं। पैसा प्राप्त करने के लिए PIN नहीं लगता।",
    phishingScam: "फिशिंग धोखाधड़ी: नकली पेमेंट लिंक/वेबसाइट आपके विवरण चुरा सकते हैं। केवल आधिकारिक लिंक खोलें।",
    fakePaymentScam: "फर्जी भुगतान: स्क्रीनशॉट पर भरोसा न करें। UPI ऐप या बैंक SMS में क्रेडिट जांचें।",
    collectScam: "कलेक्ट रिक्वेस्ट धोखाधड़ी: अज्ञात ID की रिक्वेस्ट को स्वीकार न करें।",
    otpPinScam: "OTP/PIN धोखाधड़ी: OTP और UPI PIN कभी साझा न करें।",
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
    welcome: "నేను మీ UPI Sentinel అసిస్టెంట్‌ను. డాష్‌బోర్డ్, ఫారమ్, స్కోరింగ్, సిమ్యులేషన్, భద్రతపై ఏదైనా అడగండి.",
    fallback: "నేను ఈ విషయాల్లో సహాయం చేస్తాను: ఫారమ్ ఎలా నింపాలి, స్టెప్, మొత్తం/బ్యాలెన్స్, సురక్షితం/ప్రమాదకరం/నిలిపివేసినవి, సిమ్యులేషన్, భాష మార్చడం, భద్రత సూచనలు.",
    guide: "ఈ క్రమంలో నింపండి: లావాదేవీ ఐడి, రకం, మొత్తం, స్టెప్ (1-744), పంపినవారి బ్యాలెన్స్, స్వీకరించే వారి బ్యాలెన్స్, గత 1 గంట లావాదేవీల సంఖ్య, అవసరమైతే డివైస్/లోకేషన్ మారింది ఎంపిక చేయండి.",
    tips: "OTP లేదా UPI PIN ఎప్పుడూ పంచుకోవద్దు. UPI ID మరియు పేరు ధృవీకరించండి. తెలియని QR కోడ్‌లు, నకిలీ సపోర్ట్ కాల్స్ నుండి జాగ్రత్త.",
    qrScam: "QR మోసం: డబ్బు వస్తుందని చెప్పి QR స్కాన్ చేయిస్తారు. డబ్బు స్వీకరించడానికి PIN అవసరం లేదు.",
    phishingScam: "ఫిషింగ్ మోసం: నకిలీ లింక్‌లు/పేజీలు మీ వివరాలు దొంగిలిస్తాయి. అధికారిక లింక్‌లనే వాడండి.",
    fakePaymentScam: "నకిలీ చెల్లింపు: స్క్రీన్‌షాట్ నమ్మకండి. యాప్/SMS లో నిజంగా క్రెడిట్ వచ్చిందో చూడండి.",
    collectScam: "కలెక్ట్ రిక్వెస్ట్ మోసం: తెలియని ID నుండి వచ్చిన రిక్వెస్ట్‌ను ఆమోదించవద్దు.",
    otpPinScam: "OTP/PIN మోసం: OTP, UPI PIN ఎప్పుడూ పంచుకోవద్దు.",
    step: "Step అనేది గంటల ఆధారిత టైమ్ ఇండెక్స్. 1 నుండి 744 వరకు పూర్తి సంఖ్య ఇవ్వండి.",
    amount: "Amount అంటే పంపే మొత్తం. బ్యాలెన్స్ సరిపోయేలా ఉండాలి: sender new balance సాధారణంగా old minus amount.",
    balances: "Old Balance Org/Dest = లావాదేవీకి ముందు. New Balance Org/Dest = లావాదేవీ తర్వాత.",
    device: "Device Changed ను కొత్త/అసాధారణ డివైస్ అయితే మాత్రమే ఎంచుకోండి.",
    location: "Location Changed ను అసాధారణ ప్రాంతం అయితే మాత్రమే ఎంచుకోండి.",
    txnCount: "Txns Last 1h అంటే అదే యూజర్ గత 1 గంటలో చేసిన లావాదేవీల సంఖ్య.",
    cashin: "CASH_IN అంటే డబ్బు ఖాతాలోకి రావడం. CASH_OUT అంటే డబ్బు బయటకు వెళ్లడం.",
    verdicts: "SAFE = తక్కువ ప్రమాదం, RISKY = అనుమానాస్పదం, BLOCKED = అధిక ప్రమాదం మరియు నిలిపివేత.",
    simulation: "బ్యాచ్ సిమ్యులేషన్ పెద్ద నమూనా లావాదేవీలు నడిపి సురక్షితం/ప్రమాదకరం/నిలిపివేసినవి కౌంట్ చూపిస్తుంది.",
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

function refreshVoices() {
  if (!("speechSynthesis" in window)) return;
  availableVoices = window.speechSynthesis.getVoices() || [];
}

function pickPreferredVoice() {
  const target = (SPEECH_LANG[currentLang] || "en-IN").toLowerCase();
  const langPrefix = target.slice(0, 2);
  const femaleHints = FEMALE_VOICE_HINTS[currentLang] || FEMALE_VOICE_HINTS.en;
  const nameHints = LANGUAGE_NAME_HINTS[currentLang] || [];
  const exactVoices = availableVoices.filter((v) => (v.lang || "").toLowerCase().replace("_", "-") === target);
  const exactFemale = exactVoices.find((v) => {
    const n = (v.name || "").toLowerCase();
    return femaleHints.some((hint) => n.includes(hint));
  });
  if (exactFemale) return exactFemale;
  if (exactVoices.length) return exactVoices[0];

  const langVoices = availableVoices.filter((v) => {
    const lang = (v.lang || "").toLowerCase().replace("_", "-");
    const name = (v.name || "").toLowerCase();
    return lang.startsWith(langPrefix) || nameHints.some((hint) => name.includes(hint));
  });
  if (!langVoices.length) return null;

  const femaleVoice = langVoices.find((v) => {
    const n = (v.name || "").toLowerCase();
    return femaleHints.some((hint) => n.includes(hint));
  });

  return femaleVoice || langVoices[0] || null;
}

function prepareSpeechText(text) {
  const raw = String(text || "");
  const rules = SPEECH_REWRITE_MAP[currentLang];
  if (!rules) return raw;
  return rules.reduce((acc, [pattern, replacement]) => acc.replace(pattern, replacement), raw);
}

function hasLanguageVoice(langCode) {
  const prefix = (SPEECH_LANG[langCode] || "en-IN").toLowerCase().slice(0, 2);
  const nameHints = LANGUAGE_NAME_HINTS[langCode] || [];
  return availableVoices.some((v) => {
    const lang = (v.lang || "").toLowerCase().replace("_", "-");
    const name = (v.name || "").toLowerCase();
    return lang.startsWith(prefix) || nameHints.some((hint) => name.includes(hint));
  });
}

function stopCurrentAudio() {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }
  if (currentAudioUrl) {
    URL.revokeObjectURL(currentAudioUrl);
    currentAudioUrl = null;
  }
}

function speakTextLocal(text) {
  if (!("speechSynthesis" in window)) return;
  // Voices can be empty on first call in some browsers.
  if (!availableVoices.length) refreshVoices();
  const utterance = new SpeechSynthesisUtterance(prepareSpeechText(text));
  utterance.lang = SPEECH_LANG[currentLang] || "en-IN";
  const preferredVoice = pickPreferredVoice();
  if (preferredVoice) utterance.voice = preferredVoice;
  // Softer voice profile.
  utterance.rate = 0.92;
  utterance.pitch = 1.08;
  utterance.volume = 0.9;
  stopCurrentAudio();
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

async function speakText(text) {
  const prepared = prepareSpeechText(text);
  if (!USE_CLOUD_TTS) {
    speakTextLocal(prepared);
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: prepared, language: currentLang }),
    });
    if (!res.ok) throw new Error("Cloud TTS failed");
    const blob = await res.blob();
    stopCurrentAudio();
    window.speechSynthesis.cancel();

    currentAudioUrl = URL.createObjectURL(blob);
    currentAudio = new Audio(currentAudioUrl);
    currentAudio.onended = () => stopCurrentAudio();
    currentAudio.onerror = () => stopCurrentAudio();
    await currentAudio.play();
  } catch (_err) {
    speakTextLocal(prepared);
  }
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
  document.getElementById("unknown_qr_code").checked = false;
  document.getElementById("collect_request_received").checked = false;
  document.getElementById("otp_shared").checked = false;
  document.getElementById("upi_pin_shared").checked = false;
  document.getElementById("phishing_link_clicked").checked = false;
  document.getElementById("remote_app_installed").checked = false;
  document.getElementById("screen_share_active").checked = false;
  document.getElementById("fake_payment_screenshot").checked = false;
  document.getElementById("merchant_name_mismatch").checked = false;
  document.getElementById("urgency_pressure").checked = false;
  document.getElementById("unknown_caller_request").checked = false;
  document.getElementById("suspicious_app_clone").checked = false;
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
  document.getElementById("unknown_qr_code").checked = true;
  document.getElementById("collect_request_received").checked = true;
  document.getElementById("otp_shared").checked = true;
  document.getElementById("upi_pin_shared").checked = false;
  document.getElementById("phishing_link_clicked").checked = true;
  document.getElementById("remote_app_installed").checked = false;
  document.getElementById("screen_share_active").checked = true;
  document.getElementById("fake_payment_screenshot").checked = true;
  document.getElementById("merchant_name_mismatch").checked = true;
  document.getElementById("urgency_pressure").checked = true;
  document.getElementById("unknown_caller_request").checked = true;
  document.getElementById("suspicious_app_clone").checked = false;
}

function commandMatches(command, patterns) {
  return patterns.some((pattern) => command.includes(pattern));
}

let lastAssistantIntent = "welcome";
let fallbackCursor = 0;

const FALLBACK_VARIANTS = {
  en: [
    "I can answer this better if you mention topic name like step, amount, balances, simulation, blocked, or safety tips.",
    "Ask in natural way. Example: What is step? How to fill balances? Why is a transaction blocked?",
    "I understand website questions. Try: explain this dashboard, how to score transaction, or give fraud prevention tips.",
  ],
  hi: [
    "अगर आप विषय का नाम बताएं तो मैं बेहतर जवाब दूंगा, जैसे step, amount, balance, simulation, blocked, या safety tips.",
    "सवाल सामान्य भाषा में पूछें। उदाहरण: step क्या है? balance कैसे भरें? transaction blocked क्यों हुआ?",
    "मैं वेबसाइट से जुड़े सवाल समझता हूं। पूछें: यह dashboard क्या करता है, स्कोर कैसे करें, fraud से कैसे बचें।",
  ],
  te: [
    "విషయాన్ని చెబితే నేను ఇంకా మంచి సమాధానం ఇస్తాను: step, amount, balance, simulation, blocked లేదా safety tips.",
    "సహజంగా అడగండి. ఉదా: step అంటే ఏమిటి? balances ఎలా నింపాలి? transaction ఎందుకు blocked అయింది?",
    "వెబ్‌సైట్‌కు సంబంధించిన ప్రశ్నలు నాకు అర్థమవుతాయి. అడగండి: ఈ dashboard ఏమి చేస్తుంది, score ఎలా చేయాలి, fraud నుండి ఎలా జాగ్రత్తపడాలి.",
  ],
};

function getIntentFromCommand(command) {
  const intents = [
    { id: "help", patterns: ["help", "guide", "how to fill", "form", "मदद", "सहायता", "फॉर्म", "ఎలా నింపాలి", "సహాయం", "ఫారమ్"] },
    { id: "tips", patterns: ["tips", "tip", "safety", "fraud", "otp", "सुरक्षा", "सुझाव", "टिप्स", "మోసం", "సూచనలు", "భద్రత"] },
    { id: "qrScam", patterns: ["qr scam", "unknown qr", "qr code", "क्यूआर", "qr कोड", "క్యుఆర్", "qr మోసం"] },
    { id: "phishingScam", patterns: ["phishing", "fake link", "suspicious link", "फिशिंग", "నకిలీ లింక్", "ఫిషింగ్"] },
    { id: "fakePaymentScam", patterns: ["fake payment", "screenshot", "payment screenshot", "फर्जी भुगतान", "స్క్రీన్‌షాట్", "నకిలీ చెల్లింపు"] },
    { id: "collectScam", patterns: ["collect request", "money request", "कलेक्ट रिक्वेस्ट", "కలెక్ట్ రిక్వెస్ట్"] },
    { id: "otpPinScam", patterns: ["otp", "upi pin", "pin shared", "otp shared", "ओटीपी", "upi पिन", "యుపిఐ పిన్", "otp మోసం"] },
    { id: "step", patterns: ["step", "स्टेप", "స్టెప్"] },
    { id: "amount", patterns: ["amount", "राशि", "amount क्या", "మొత్తం", "ఎంత మొత్తం"] },
    { id: "balances", patterns: ["balance", "old balance", "new balance", "बैलेंस", "शेष", "బ్యాలెన్స్"] },
    { id: "device", patterns: ["device changed", "device", "डिवाइस", "డివైస్"] },
    { id: "location", patterns: ["location changed", "location", "लोकेशन", "ప్రాంతం", "లోకేషన్"] },
    { id: "txnCount", patterns: ["txns", "last 1h", "1 hour", "1 घंटे", "1 గంట", "recent transactions"] },
    { id: "cashin", patterns: ["cash in", "cash-out", "cash out", "cashin", "cash_in", "cash_out"] },
    { id: "verdicts", patterns: ["verdict", "blocked", "risky", "safe", "निर्णय", "ब्लॉक", "ప్రమాదకరం", "నిలిపివేసినవి"] },
    { id: "backend", patterns: ["backend", "server", "health", "api"] },
    { id: "simulation", patterns: ["simulate", "batch", "simulation", "सिमुलेशन", "बैच", "సిమ్యులేషన్", "బ్యాచ్"] },
    { id: "score", patterns: ["score", "submit", "check transaction", "स्कोर", "सबमिट", "స్కోర్"] },
    { id: "safeExample", patterns: ["safe example", "safe form", "सुरक्षित उदाहरण", "సురక్షిత ఉదాహరణ"] },
    { id: "riskyExample", patterns: ["risky example", "fraud example", "जोखिम उदाहरण", "ప్రమాద ఉదాహరణ"] },
    { id: "clear", patterns: ["clear", "reset", "रीसेट", "రిసెట్"] },
    { id: "stop", patterns: ["stop", "cancel voice", "वॉइस बंद", "వాయిస్ ఆపు", "ఆపు"] },
    { id: "website", patterns: ["website", "dashboard", "what this", "about this", "इसके बारे", "बारे में", "దీని గురించి", "dashboard గురించి"] },
  ];

  const scores = intents.map((intent) => {
    let score = 0;
    for (const p of intent.patterns) {
      if (command.includes(p)) score += 1 + (p.length > 6 ? 0.25 : 0);
    }
    return { id: intent.id, score };
  });

  scores.sort((a, b) => b.score - a.score);
  return scores[0].score > 0 ? scores[0].id : null;
}

function runAssistantCommand(rawCommand) {
  const command = (rawCommand || "").trim().toLowerCase();
  if (!command) return;
  addChatMessage(rawCommand, "user");

  let response = "";
  const languagePatterns = ["language", "hindi", "english", "telugu", "भाषा", "భాష", "हिंदी", "తెలుగు", "ఇంగ్లీష్"];
  const languageSwitchVerbs = ["change", "switch", "to ", "बदल", "మార్చ", "set"];
  const intent = getIntentFromCommand(command);

  if (commandMatches(command, languagePatterns) && commandMatches(command, languageSwitchVerbs)) {
    if (command.includes("hindi") || command.includes("hi")) refs.language.value = "hi";
    else if (command.includes("telugu") || command.includes("te")) refs.language.value = "te";
    else if (command.includes("english") || command.includes("en")) refs.language.value = "en";
    else if (command.includes("हिंदी")) refs.language.value = "hi";
    else if (command.includes("తెలుగు")) refs.language.value = "te";
    else if (command.includes("इंग्लिश") || command.includes("ఆంగ్ల")) refs.language.value = "en";
    applyLanguage(refs.language.value);
    response = formatMessage(kb("langChanged"), { lang: LANGUAGE_LABELS[refs.language.value] });
    lastAssistantIntent = "language";
  } else if (intent === "help") {
    response = kb("guide");
    lastAssistantIntent = "help";
  } else if (intent === "tips") {
    response = kb("tips");
    lastAssistantIntent = "tips";
  } else if (intent === "qrScam") {
    response = kb("qrScam");
    lastAssistantIntent = "qrScam";
  } else if (intent === "phishingScam") {
    response = kb("phishingScam");
    lastAssistantIntent = "phishingScam";
  } else if (intent === "fakePaymentScam") {
    response = kb("fakePaymentScam");
    lastAssistantIntent = "fakePaymentScam";
  } else if (intent === "collectScam") {
    response = kb("collectScam");
    lastAssistantIntent = "collectScam";
  } else if (intent === "otpPinScam") {
    response = kb("otpPinScam");
    lastAssistantIntent = "otpPinScam";
  } else if (intent === "step") {
    response = kb("step");
    lastAssistantIntent = "step";
  } else if (intent === "amount") {
    response = kb("amount");
    lastAssistantIntent = "amount";
  } else if (intent === "balances") {
    response = kb("balances");
    lastAssistantIntent = "balances";
  } else if (intent === "device") {
    response = kb("device");
    lastAssistantIntent = "device";
  } else if (intent === "location") {
    response = kb("location");
    lastAssistantIntent = "location";
  } else if (intent === "txnCount") {
    response = kb("txnCount");
    lastAssistantIntent = "txnCount";
  } else if (intent === "cashin") {
    response = kb("cashin");
    lastAssistantIntent = "cashin";
  } else if (intent === "verdicts") {
    response = kb("verdicts");
    lastAssistantIntent = "verdicts";
  } else if (intent === "backend") {
    response = kb("backend");
    lastAssistantIntent = "backend";
  } else if (intent === "simulation") {
    response = kb("simStart");
    refs.simulateBtn.click();
    lastAssistantIntent = "simulation";
  } else if (intent === "score") {
    response = kb("scored");
    refs.form.requestSubmit();
    lastAssistantIntent = "score";
  } else if (intent === "safeExample") {
    fillSafeExample();
    response = kb("filledSafe");
    lastAssistantIntent = "safeExample";
  } else if (intent === "riskyExample") {
    fillRiskyExample();
    response = kb("filledRisky");
    lastAssistantIntent = "riskyExample";
  } else if (intent === "clear") {
    refs.form.reset();
    document.getElementById("recent_txn_count_1h").value = "0";
    response = kb("guide");
    lastAssistantIntent = "clear";
  } else if (intent === "stop") {
    stopVoiceAssistant();
    return;
  } else if (intent === "website" || command.includes("what") || command.includes("website") || command.includes("dashboard")) {
    response = kb("welcome");
    lastAssistantIntent = "website";
  } else if ((command.includes("its") || command.includes("इसके") || command.includes("దాని")) && lastAssistantIntent) {
    response = kb(lastAssistantIntent) || kb("fallback");
  } else {
    const variants = FALLBACK_VARIANTS[currentLang] || FALLBACK_VARIANTS.en;
    response = variants[fallbackCursor % variants.length];
    fallbackCursor += 1;
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
  stopCurrentAudio();
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

function formatFraudCategories(categories) {
  if (!categories || !categories.length) return "None";
  return categories.join(", ");
}

function addResultRowValues(transactionId, type, amount, verdict, riskScore, tip, fraudCategories) {
  const row = document.createElement("tr");
  row.innerHTML = `
    <td>${transactionId}</td>
    <td>${type}</td>
    <td>${Number(amount).toFixed(2)}</td>
    <td class="${statusClass(verdict)}">${verdict}</td>
    <td>${riskScore}%</td>
    <td>${formatFraudCategories(fraudCategories)}</td>
    <td>${tip}</td>
  `;
  refs.table.prepend(row);

  while (refs.table.children.length > 25) {
    refs.table.removeChild(refs.table.lastChild);
  }
}

function addResultRow(payload, result) {
  addResultRowValues(
    result.transaction_id,
    payload.type,
    payload.amount,
    result.verdict,
    result.risk_score,
    result.tip,
    result.fraud_categories,
  );
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
    "assistantAskBtn", "autoVoiceLabel", "scenarioTransaction", "scenarioQrScam", "scenarioPhishing",
    "scenarioFakePay", "scenarioOtpPin", "scenarioCollect", "scenarioRemote", "scenarioCaller",
    "scenarioFakeApp", "applyScenarioBtn",
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
  refs.scenarioHint.textContent = t("scenarioHint_default");
}

const NUMBER_DEFAULTS = {
  amount: 500,
  step: 120,
  oldbalanceOrg: 10000,
  newbalanceOrig: 9500,
  oldbalanceDest: 20000,
  newbalanceDest: 20500,
  recent_txn_count_1h: 1,
};

const BASIC_FIELDS = [
  "transaction_id", "type", "amount", "step", "oldbalanceOrg", "newbalanceOrig",
  "oldbalanceDest", "newbalanceDest", "recent_txn_count_1h", "device_changed", "location_changed",
];

const SCENARIO_RULES = {
  transaction_full: {
    show: BASIC_FIELDS,
    required: ["transaction_id", "type", "amount", "step", "oldbalanceOrg", "newbalanceOrig"],
    showSignals: [
      "unknown_qr_code", "collect_request_received", "otp_shared", "upi_pin_shared", "phishing_link_clicked",
      "remote_app_installed", "screen_share_active", "fake_payment_screenshot", "merchant_name_mismatch",
      "urgency_pressure", "unknown_caller_request", "suspicious_app_clone",
    ],
    autoSignals: [],
  },
  qr_scam: {
    show: ["transaction_id", "type", "amount", "step", "merchant_name_mismatch"],
    required: ["transaction_id", "type", "amount"],
    showSignals: ["merchant_name_mismatch", "urgency_pressure", "unknown_caller_request"],
    autoSignals: [],
  },
  phishing_link: {
    show: ["transaction_id", "type", "amount", "device_changed", "location_changed"],
    required: ["transaction_id", "type", "amount"],
    showSignals: ["phishing_link_clicked", "unknown_caller_request", "urgency_pressure"],
    autoSignals: ["phishing_link_clicked"],
  },
  fake_payment: {
    show: ["transaction_id", "type", "amount", "oldbalanceDest", "newbalanceDest"],
    required: ["transaction_id", "type", "amount"],
    showSignals: ["fake_payment_screenshot", "merchant_name_mismatch"],
    autoSignals: ["fake_payment_screenshot"],
  },
  otp_pin: {
    show: ["transaction_id", "type", "amount"],
    required: ["transaction_id", "type", "amount"],
    showSignals: ["otp_shared", "upi_pin_shared", "unknown_caller_request", "urgency_pressure"],
    autoSignals: ["otp_shared", "unknown_caller_request"],
  },
  collect_request: {
    show: ["transaction_id", "type", "amount", "step"],
    required: ["transaction_id", "type", "amount"],
    showSignals: ["collect_request_received", "unknown_caller_request", "merchant_name_mismatch"],
    autoSignals: ["collect_request_received"],
  },
  remote_access: {
    show: ["transaction_id", "type", "amount", "device_changed"],
    required: ["transaction_id", "type", "amount"],
    showSignals: ["remote_app_installed", "screen_share_active", "otp_shared", "upi_pin_shared"],
    autoSignals: ["remote_app_installed"],
  },
  caller_pressure: {
    show: ["transaction_id", "type", "amount"],
    required: ["transaction_id", "type", "amount"],
    showSignals: ["unknown_caller_request", "urgency_pressure", "otp_shared", "upi_pin_shared"],
    autoSignals: ["unknown_caller_request", "urgency_pressure"],
  },
  fake_app: {
    show: ["transaction_id", "type", "amount", "device_changed"],
    required: ["transaction_id", "type", "amount"],
    showSignals: ["suspicious_app_clone", "otp_shared", "upi_pin_shared"],
    autoSignals: ["suspicious_app_clone"],
  },
};

function toggleField(id, visible) {
  const el = document.getElementById(id);
  if (!el) return;
  const tag = el.tagName.toLowerCase();
  const wrapper = tag === "input" && el.type === "checkbox" ? el.closest("label") : el;
  if (wrapper) wrapper.style.display = visible ? "" : "none";
}

function applyScenario() {
  const key = refs.fraudScenario.value;
  const rule = SCENARIO_RULES[key] || SCENARIO_RULES.transaction_full;
  const allSignals = SCENARIO_RULES.transaction_full.showSignals;

  for (const fieldId of BASIC_FIELDS) {
    const visible = rule.show.includes(fieldId);
    toggleField(fieldId, visible);
    const el = document.getElementById(fieldId);
    if (!el) continue;
    if ("required" in el) el.required = visible && rule.required.includes(fieldId);
    if (!visible && el.tagName.toLowerCase() === "input" && el.type !== "checkbox" && NUMBER_DEFAULTS[fieldId] !== undefined) {
      el.value = NUMBER_DEFAULTS[fieldId];
    }
    if (!visible && el.type === "checkbox") el.checked = false;
  }

  for (const signalId of allSignals) {
    const el = document.getElementById(signalId);
    if (!el) continue;
    const visible = rule.showSignals.includes(signalId);
    toggleField(signalId, visible);
    el.checked = rule.autoSignals.includes(signalId);
  }

  if (!document.getElementById("transaction_id").value) {
    document.getElementById("transaction_id").value = `TXN-${Date.now().toString().slice(-6)}`;
  }
  if (key !== "qr_scam") {
    document.getElementById("q_unknown_qr").checked = false;
    document.getElementById("q_scan_receive_money").checked = false;
    document.getElementById("q_qr_pin_otp_prompt").checked = false;
  }
  if (key !== "fake_app") {
    document.getElementById("q_fakeapp_otp_requested").checked = false;
  }
  refs.scenarioHint.textContent = `${t("applyScenarioBtn")}: ${document.querySelector("#fraudScenario option:checked")?.textContent || ""}`;
  refs.fraudSignalsPanel.style.display = key === "transaction_full" ? "" : "none";
  refs.scenarioQuestions.style.display = key === "qr_scam" ? "" : "none";
  refs.scenarioQuestionsFakeApp.style.display = key === "fake_app" ? "" : "none";
  if (key !== "transaction_full") {
    refs.scenarioHint.textContent = `${refs.scenarioHint.textContent} | Fraud signals are auto-set for this fraud type.`;
  }
}

function formPayload() {
  const isQrScenario = refs.fraudScenario.value === "qr_scam";
  const isFakeAppScenario = refs.fraudScenario.value === "fake_app";
  const qrUnknown = isQrScenario ? document.getElementById("q_unknown_qr").checked : document.getElementById("unknown_qr_code").checked;
  const qrReceive = isQrScenario ? document.getElementById("q_scan_receive_money").checked : false;
  const qrPinPrompt = isQrScenario ? document.getElementById("q_qr_pin_otp_prompt").checked : false;
  const fakeAppOtp = isFakeAppScenario ? document.getElementById("q_fakeapp_otp_requested").checked : document.getElementById("otp_shared").checked;

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
    unknown_qr_code: qrUnknown,
    asked_scan_to_receive_money: qrReceive,
    pin_or_otp_prompt_after_qr: qrPinPrompt,
    collect_request_received: document.getElementById("collect_request_received").checked,
    otp_shared: fakeAppOtp,
    upi_pin_shared: isFakeAppScenario ? fakeAppOtp : document.getElementById("upi_pin_shared").checked,
    phishing_link_clicked: document.getElementById("phishing_link_clicked").checked,
    remote_app_installed: document.getElementById("remote_app_installed").checked,
    screen_share_active: document.getElementById("screen_share_active").checked,
    fake_payment_screenshot: document.getElementById("fake_payment_screenshot").checked,
    merchant_name_mismatch: document.getElementById("merchant_name_mismatch").checked,
    urgency_pressure: document.getElementById("urgency_pressure").checked,
    unknown_caller_request: document.getElementById("unknown_caller_request").checked,
    suspicious_app_clone: document.getElementById("suspicious_app_clone").checked,
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
    const fraudInfo = formatFraudCategories(result.fraud_categories);
    refs.result.textContent = `${formatMessage(t("resultFormat"), {
      verdict: result.verdict,
      risk: result.risk_score,
      lang: activeLanguage,
      tip: result.tip,
    })} | Fraud Types: ${fraudInfo}`;
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
      addResultRowValues(
        row.transaction_id,
        row.type,
        row.amount,
        row.verdict,
        row.risk_score,
        row.tip,
        row.fraud_categories,
      );
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

refs.applyScenarioBtn.addEventListener("click", () => {
  applyScenario();
});

refs.fraudScenario.addEventListener("change", () => {
  applyScenario();
});

refs.language.addEventListener("change", () => {
  applyLanguage(refs.language.value);
  checkBackend();
  const langText = formatMessage(t("languageChanged"), {
    lang: LANGUAGE_LABELS[refs.language.value] || refs.language.value,
  });
  refs.result.textContent = langText;
  addChatMessage(langText, "bot");
  if ((refs.language.value === "hi" || refs.language.value === "te") && !hasLanguageVoice(refs.language.value)) {
    addChatMessage("Hindi/Telugu voice is not installed in this browser/OS. It may read using another voice.", "bot");
  }
  setVoiceStatus(t("voiceReady"));
  updateSummary();
});

refs.openComplaintPortalBtn?.addEventListener("click", async () => {
  const incidentType = refs.complaintType?.value || "UPI Fraud";
  const details = (refs.complaintDetails?.value || "").trim();
  const txId = document.getElementById("transaction_id")?.value || "N/A";
  const amount = document.getElementById("amount")?.value || "N/A";
  const summary = [
    `Incident Type: ${incidentType}`,
    `Transaction ID: ${txId}`,
    `Amount: ${amount}`,
    `Scenario: ${refs.fraudScenario?.value || "N/A"}`,
    `Details: ${details || "N/A"}`,
  ].join("\n");

  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(summary);
      refs.complaintStatus.textContent = "Complaint summary copied. Please paste it on cybercrime portal form.";
    } else {
      refs.complaintStatus.textContent = "Clipboard not available. Please manually copy details and submit on portal.";
    }
  } catch (_err) {
    refs.complaintStatus.textContent = "Could not auto-copy summary. You can still file complaint on portal.";
  }

  window.open("https://cybercrime.gov.in/Webform/Accept.aspx", "_blank", "noopener,noreferrer");
});

document.getElementById("navLogout")?.addEventListener("click", (event) => {
  event.preventDefault();
  localStorage.removeItem(AUTH_KEY);
  window.location.href = "login.html";
});

applyLanguage(refs.language.value);
applyScenario();
refs.result.textContent = t("defaultResult");
refs.apiState.textContent = t("checkingBackend");
setVoiceStatus(t("voiceReady"));
addChatMessage(kb("welcome"), "bot");
refreshVoices();
if ("speechSynthesis" in window) {
  window.speechSynthesis.onvoiceschanged = refreshVoices;
}
checkBackend();
updateSummary();
