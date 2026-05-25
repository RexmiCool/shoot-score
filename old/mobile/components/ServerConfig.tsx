/**
 * Composant : configuration de l'URL du serveur.
 * Affiché dans un modal depuis l'écran Home.
 */

import React, { useEffect, useState } from "react";
import {
  ActivityIndicator,
  Modal,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { Colors } from "../constants/Colors";
import { checkHealth } from "../services/api";
import { getServerUrl, setServerUrl } from "../services/storage";

interface Props {
  visible: boolean;
  onClose: () => void;
}

export default function ServerConfig({ visible, onClose }: Props) {
  const [url,       setUrl]       = useState("");
  const [status,    setStatus]    = useState<"idle" | "checking" | "ok" | "error">("idle");
  const [errorMsg,  setErrorMsg]  = useState<string | null>(null);

  useEffect(() => {
    if (visible) {
      getServerUrl().then(setUrl);
      setStatus("idle");
    }
  }, [visible]);

  async function handleSave() {
    setErrorMsg(null);
    await setServerUrl(url);
    setStatus("checking");
    const err = await checkHealth();
    if (err === null) {
      setStatus("ok");
    } else {
      setStatus("error");
      setErrorMsg(err);
    }
  }

  const statusColor = status === "ok"
    ? Colors.success
    : status === "error"
    ? Colors.error
    : Colors.textSecondary;

  const statusText = {
    idle:     "",
    checking: "Vérification…",
    ok:       "✓ Serveur accessible",
    error:    errorMsg ? `✗ ${errorMsg}` : "✗ Serveur inaccessible",
  }[status];

  return (
    <Modal
      visible={visible}
      transparent
      animationType="slide"
      onRequestClose={onClose}
    >
      <Pressable style={styles.overlay} onPress={onClose}>
        <Pressable style={styles.sheet} onPress={() => {}}>
          <Text style={styles.title}>⚙️  Serveur</Text>
          <Text style={styles.hint}>
            Adresse IP du PC qui fait tourner l'API (même réseau Wi-Fi).{"\n"}
            Lancer sur le PC :{"\n"}
            <Text style={styles.code}>
              uvicorn src.api:app --host 0.0.0.0 --port 8000
            </Text>
          </Text>

          <TextInput
            style={styles.input}
            value={url}
            onChangeText={setUrl}
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="url"
            placeholder="http://192.168.1.x:8000"
            placeholderTextColor={Colors.textMuted}
          />

          <View style={styles.actions}>
            <Pressable style={styles.btnSecondary} onPress={onClose}>
              <Text style={styles.btnSecondaryText}>Annuler</Text>
            </Pressable>
            <Pressable style={styles.btnPrimary} onPress={handleSave}>
              {status === "checking" ? (
                <ActivityIndicator color="#000" size="small" />
              ) : (
                <Text style={styles.btnPrimaryText}>Enregistrer et tester</Text>
              )}
            </Pressable>
          </View>

          {statusText ? (
            <Text style={[styles.statusText, { color: statusColor }]}>
              {statusText}
            </Text>
          ) : null}
        </Pressable>
      </Pressable>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: "#000000AA",
    justifyContent: "flex-end",
  },
  sheet: {
    backgroundColor: Colors.surface,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: 24,
    gap: 14,
  },
  title: {
    color: Colors.textPrimary,
    fontSize: 18,
    fontWeight: "700",
  },
  hint: {
    color: Colors.textSecondary,
    fontSize: 13,
    lineHeight: 20,
  },
  code: {
    fontFamily: "monospace",
    color: Colors.primary,
    fontSize: 12,
  },
  input: {
    backgroundColor: Colors.surfaceElevated,
    borderColor: Colors.border,
    borderWidth: 1,
    borderRadius: 10,
    padding: 12,
    color: Colors.textPrimary,
    fontSize: 15,
    fontFamily: "monospace",
  },
  actions: {
    flexDirection: "row",
    gap: 10,
  },
  btnPrimary: {
    flex: 1,
    backgroundColor: Colors.primary,
    borderRadius: 10,
    padding: 14,
    alignItems: "center",
  },
  btnPrimaryText: { color: "#000", fontWeight: "700", fontSize: 15 },
  btnSecondary: {
    flex: 0,
    paddingHorizontal: 18,
    borderRadius: 10,
    padding: 14,
    backgroundColor: Colors.surfaceElevated,
    alignItems: "center",
  },
  btnSecondaryText: { color: Colors.textSecondary, fontSize: 15 },
  statusText: {
    textAlign: "center",
    fontSize: 13,
    fontWeight: "600",
  },
});
