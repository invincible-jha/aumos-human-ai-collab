/**
 * AumOS HITL Review Interface — Keyboard Shortcuts and Form Helpers
 *
 * Keyboard shortcuts:
 *   Ctrl+A — Approve (sets decision select to "approved" and submits)
 *   Ctrl+R — Reject  (sets decision select to "rejected" and submits)
 */

(function () {
    "use strict";

    function submitDecision(decision) {
        var decisionSelect = document.getElementById("decision");
        var form = document.getElementById("decision-form");
        var justificationField = document.getElementById("justification");

        if (!decisionSelect || !form) return;

        decisionSelect.value = decision;

        // Dispatch change event to update modifications field visibility
        decisionSelect.dispatchEvent(new Event("change"));

        // Require justification — focus if empty
        if (!justificationField.value.trim()) {
            justificationField.focus();
            justificationField.placeholder = "Justification is required before submitting.";
            return;
        }

        form.submit();
    }

    document.addEventListener("keydown", function (event) {
        // Only handle shortcuts if no input/textarea has focus
        var active = document.activeElement;
        var isInputActive =
            active &&
            (active.tagName === "TEXTAREA" || active.tagName === "INPUT" || active.tagName === "SELECT");

        if (!isInputActive && event.ctrlKey && !event.shiftKey && !event.altKey) {
            if (event.key === "a" || event.key === "A") {
                event.preventDefault();
                submitDecision("approved");
            } else if (event.key === "r" || event.key === "R") {
                event.preventDefault();
                submitDecision("rejected");
            }
        }
    });

    // Prevent double-submit on form submit button
    var submitBtn = document.getElementById("submit-btn");
    if (submitBtn) {
        var form = document.getElementById("decision-form");
        if (form) {
            form.addEventListener("submit", function () {
                submitBtn.disabled = true;
                submitBtn.textContent = "Submitting...";
            });
        }
    }
})();
