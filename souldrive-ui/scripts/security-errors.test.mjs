import assert from "node:assert/strict";
import { test } from "node:test";

import {
  formatSecurityActionError,
  securityPanelMode,
} from "../../build/security-errors-test/securityErrors.js";

test("securityPanelMode treats missing security status as unavailable", () => {
  assert.equal(securityPanelMode(null), "unavailable");
});

test("formatSecurityActionError explains an outdated sidecar on init 404", () => {
  assert.equal(
    formatSecurityActionError("init", 404),
    "当前本地 sidecar 版本不支持工作区解锁，请重新打包或更新本地服务。",
  );
});

test("formatSecurityActionError asks user to unlock when init sees an existing keystore", () => {
  assert.equal(
    formatSecurityActionError("init", 409, "workspace keystore already initialized"),
    "该工作区已经初始化，请直接输入口令解锁。",
  );
});

test("formatSecurityActionError gives a direct password error on unlock 403", () => {
  assert.equal(
    formatSecurityActionError("unlock", 403, "incorrect passphrase"),
    "口令错误，请重新输入。",
  );
});
